from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

import pyarrow  # type: ignore
from pyarrow import parquet as pq

import opencosmo as oc

if TYPE_CHECKING:
    from os import PathLike

    from opencosmo.collection.protocols import Collection


def write_parquet(
    path: PathLike,
    to_write: oc.Dataset | Collection,
    overwrite: bool = False,
    **kwargs,
):
    """
    Write a dataset or collection to a parquet file at the given path. If you are writing a :py:class:`opencosmo.Dataset`,
    or :py:class:`opencosmo.Lightcone` the data will be written as a single file at the given path. If you are writing
    a :py:class:`opencosmo.StructureCollection` the data will be written to several files, one for each type of particle.

    Parameters
    ----------
    path: PathLike
        The path to write the data to. If you are writing a :py:class:`Dataset  <opencosmo.Dataset>`
        or :py:class:`Lightcone <opencosmo.Lightcone>` this should be a single parquet file. If you
        are writing a :py:class:`StructureCollection <opencosmo.StructureCollection>`, this should be
        a folder.

    to_write: opencosmo.Dataset | opencosmo.Lightcone | opencosmo.StructureCollection
        The dataset or collection to write

    overwrite: bool, default = False
        If true, any existing data at the given path will be overwritten.


    """
    if not isinstance(path, Path):
        path = Path(path)
    match to_write:
        case oc.Dataset() | oc.Lightcone():
            return __write_dataset(path, to_write, overwrite, **kwargs)
        case oc.StructureCollection():
            return __write_structure_collection(path, to_write, overwrite, **kwargs)
        case _:
            raise ValueError(f"No parquet writer defined for type {type(to_write)}")


def __write_dataset(path: Path, dataset: oc.Dataset | oc.Lightcone, overwrite: bool):
    data = dataset.get_data("numpy")
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    output = {}
    if not isinstance(data, dict):
        raise NotImplementedError
    for name, column in data.items():
        if column.dtype.names is None:
            output[name] = column
            continue
        outputs = {f"{name}_{cname}": column[cname] for cname in column.dtype.names}
        output.update(outputs)

    table = pyarrow.table(output)
    pq.write_table(table, path)


def __write_structure_collection(
    path: Path, collection: oc.StructureCollection, overwrite: bool
):
    if path.exists() and not path.is_dir():
        raise ValueError(
            "Dumping a structure collection to parquet can create multiple files, so the output location should be a directory"
        )

    elif not path.exists():
        path.mkdir(parents=True)

    for name, dataset in collection.items():
        if "particle" not in name:
            warn(f"write_parquet only supports writing particles, skipping {name}")
            continue
        assert isinstance(dataset, oc.Dataset)
        __write_dataset(path / f"{name}.parquet", dataset, overwrite)
