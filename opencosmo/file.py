from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Concatenate, TypeVar

import h5py

from opencosmo.mpi import get_comm_world

H5Resource = TypeVar("H5Resource", h5py.File, h5py.Group, h5py.Dataset)
H5Reader = Callable[Concatenate[H5Resource, ...], Any]
FileReader = Callable[Concatenate[h5py.File | h5py.Group, ...], Any]
FileWriter = Callable[Concatenate[h5py.File | h5py.Group, ...], None]

"""
A collection of utilities for working with hdf5 files.
"""


class FileExistance(Enum):
    MUST_EXIST = "must_exist"
    MUST_NOT_EXIST = "must_not_exist"
    EITHER = "either"


def file_reader(func: FileReader) -> FileReader:
    """
    Resolves the path to a given file and handles opening
    and closing the file for reading purposes. Used as a
    decorator.
    """

    @wraps(func)
    def wrapper(file: h5py.File | h5py.Group | Path | str, *args, **kwargs):
        if not isinstance(file, (h5py.File, h5py.Group)):
            path = resolve_path(file, FileExistance.MUST_EXIST)
            with h5py.File(path, "r", driver="core") as f:
                return func(f, *args, **kwargs)
        return func(file, *args, **kwargs)

    return wrapper


def file_writer(func: FileWriter) -> FileWriter:
    """
    Resolves the path to a given file and handles opening
    and closing the file for writing purposes. Used as a
    decorator.
    """

    @wraps(func)
    def wrapper(file: h5py.File | Path | str, *args, **kwargs):
        if not isinstance(file, h5py.File | h5py.Group):
            file = resolve_path(file, FileExistance.MUST_NOT_EXIST)
        return func(file, *args, **kwargs)

    return wrapper


def broadcast_read(func: H5Reader) -> H5Reader:
    """
    If MPI is available, the decorated function will only
    be called on rank 0, with the results broadcast to all
    ranks. When we read actual data this is not necessary,
    because the data will be chunked by rank, but for attributes
    its better to not try to read them from a hundred processes
    at the same time.
    """

    @wraps(func)
    def wrapper(file: h5py.File | Path | str, *args, **kwargs):
        output = None
        comm = get_comm_world()
        if comm is None or comm.Get_rank() == 0:
            try:
                output = func(file, *args, **kwargs)
            except Exception as e:
                output = e
        if comm is not None:
            # Broadcasting the error ensures other ranks
            # will raise the exception and quit.
            output = comm.bcast(output, root=0)
        if isinstance(output, Exception):
            raise output
        return output

    return wrapper


def get_data_structure(group: h5py.Group) -> dict[str, Any]:
    units = {name: group[name].attrs.get("unit", "") for name in group.keys()}
    return units


def resolve_path(
    path: Path | str, existance: FileExistance = FileExistance.MUST_EXIST
) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if not path.exists() and existance == FileExistance.MUST_EXIST:
        raise FileNotFoundError(f"{path} does not exist.")
    if path.exists() and existance == FileExistance.MUST_NOT_EXIST:
        raise FileExistsError(f"{path} already exists.")
    if not path.parent.exists():
        raise FileNotFoundError(f"Directory {path.parent} does not exist!")
    if path.suffix != ".hdf5":
        raise ValueError(f"{path} does not appear to be an hdf5 file.")
    return path
