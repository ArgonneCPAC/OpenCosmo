from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Concatenate

import h5py

try:
    import mpi4py.MPI as MPI
except ImportError:
    MPI = None  # type: ignore

FileReader = Callable[Concatenate[h5py.File, ...], Any]
FileWriter = Callable[Concatenate[h5py.File, ...], None]


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
    def wrapper(file: h5py.File | Path | str, *args, **kwargs):
        if not isinstance(file, h5py.File):
            path = resolve_path(file, FileExistance.MUST_EXIST)
            with h5py.File(path, "r") as f:
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
        if not isinstance(file, h5py.File):
            path = resolve_path(file, FileExistance.MUST_NOT_EXIST)
            if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
                MPI.COMM_WORLD.barrier()
                with h5py.File(path, "w", driver="mpio", comm=MPI.COMM_WORLD) as f:
                    return func(f, *args, **kwargs)

            with h5py.File(path, "w") as f:
                return func(f, *args, **kwargs)

        return func(file, *args, **kwargs)

    return wrapper


def broadcast_read(func: FileReader) -> FileReader:
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
        if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
            output = func(file, *args, **kwargs)
        if MPI is not None:
            output = MPI.COMM_WORLD.bcast(output, root=0)
        return output

    return wrapper


def resolve_path(
    path: Path | str, existance: FileExistance = FileExistance.MUST_EXIST
) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if not path.exists() and existance == FileExistance.MUST_EXIST:
        raise FileNotFoundError(f"{path} does not exist.")
    if path.exists() and existance == FileExistance.MUST_NOT_EXIST:
        raise FileExistsError(f"{path} already exists.")
    if path.suffix != ".hdf5":
        raise ValueError(f"{path} does not appear to be an hdf5 file.")
    return path
