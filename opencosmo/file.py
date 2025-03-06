from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Concatenate

import h5py

FileReader = Callable[Concatenate[h5py.File, ...], Any]
FileWriter = Callable[Concatenate[h5py.File, ...], None]


class FileExistance(Enum):
    MUST_EXIST = "must_exist"
    MUST_NOT_EXIST = "must_not_exist"
    EITHER = "either"


def file_reader(func: FileReader) -> FileReader:
    @wraps(func)
    def wrapper(file: h5py.File | Path | str, *args, **kwargs):
        if not isinstance(file, h5py.File):
            path = resolve_path(file, FileExistance.MUST_EXIST)
            with h5py.File(path, "r") as f:
                return func(f, *args, **kwargs)
        return func(file, *args, **kwargs)

    return wrapper


def file_writer(func: FileWriter) -> FileWriter:
    @wraps(func)
    def wrapper(file: h5py.File | Path | str, *args, **kwargs):
        if not isinstance(file, h5py.File):
            path = resolve_path(file, FileExistance.MUST_NOT_EXIST)
            with h5py.File(path, "w") as f:
                return func(f, *args, **kwargs)
        return func(file, *args, **kwargs)

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
