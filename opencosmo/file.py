from functools import singledispatch, wraps
from pathlib import Path
from typing import Any, Callable, Concatenate

import h5py

FileReader = Callable[Concatenate[h5py.File, ...], Any]


def oc_reader(func: FileReader) -> FileReader:
    @wraps(func)
    def wrapper(file: h5py.File | Path | str, *args, **kwargs):
        return read_file(file, func, *args, **kwargs)

    return wrapper


@singledispatch
def read_file(file: h5py.File, reader: FileReader, *args, **kwargs) -> Any:
    return reader(file, *args, **kwargs)


@read_file.register
def _(path: Path, reader: FileReader, *args, **kwargs) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    if not path.is_file() or path.suffix != ".hdf5":
        raise ValueError(f"{path} does not appear to be an hdf5 file.")
    with h5py.File(path, "r") as f:
        return read_file(f, reader, *args, **kwargs)


@read_file.register
def _(path: str, reader: FileReader, *args, **kwargs) -> Any:
    return read_file(Path(path), reader, *args, **kwargs)
