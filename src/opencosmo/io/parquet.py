from os import PathLike
from pathlib import Path
from warnings import warn

import pyarrow
from pyarrow import parquet as pq

import opencosmo as oc
from opencosmo.collection.protocols import Collection


def write_parquet(path: PathLike, to_write: oc.Dataset | Collection, *args, **kwargs):
    if not isinstance(path, Path):
        path = Path(path)
    match to_write:
        case oc.Dataset() | oc.Lightcone():
            return __write_dataset(path, to_write, *args, **kwargs)
        case oc.StructureCollection():
            return __write_structure_collection(path, to_write, *args, **kwargs)
        case _:
            raise ValueError(f"No parquet writer defined for type {type(to_write)}")


def __write_dataset(path: Path, dataset: oc.Dataset | oc.Lightcone):
    data = dataset.get_data("numpy")
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


def __write_structure_collection(path: Path, collection: oc.StructureCollection):
    if len(collection) > 1:
        raise ValueError(
            "write_parquet currently only supports writing a single structure"
        )

    for name, dataset in collection.items():
        if "particle" not in name:
            warn(f"write_parquet only supports writing particles, skipping {name}")
            continue
        assert isinstance(dataset, oc.Dataset)
        __write_dataset(path / f"{name}.parquet", dataset)
