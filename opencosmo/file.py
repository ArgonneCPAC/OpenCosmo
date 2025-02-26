from functools import wraps
from pathlib import Path
from typing import Any, Callable, Concatenate

import h5py

FileReader = Callable[Concatenate[h5py.File, ...], Any]


def file_reader(func: FileReader) -> FileReader:
    @wraps(func)
    def wrapper(file: h5py.File | Path | str, *args, **kwargs):
        if not isinstance(file, h5py.File):
            path = resolve_path(file)
            with h5py.File(path, "r") as f:
                return func(f, *args, **kwargs)
        return func(file, *args, **kwargs)

    return wrapper


def resolve_path(path: Path | str) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    if not path.is_file() or path.suffix != ".hdf5":
        raise ValueError(f"{path} does not appear to be an hdf5 file.")
    return path
