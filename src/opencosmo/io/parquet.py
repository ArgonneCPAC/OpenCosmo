from os import PathLike
from warnings import warn

import numpy as np
import pyarrow

import opencosmo as oc
from opencosmo.collection.protocols import Collection


def write_parquet(path: PathLike, to_write: oc.Dataset | Collection, *args, **kwargs):
    match to_write:
        case oc.Dataset() | oc.Lightcone():
            return __write_dataset(path, to_write, *args, **kwargs)
        case oc.StructureCollection():
            return __write_structure_collection(path, to_write, *args, **kwargs)


def __write_dataset(path: PathLike, dataset: oc.Dataset | oc.Lightcone):
    data = dataset.get_data("numpy")
    table = pyarrow.table(data)
    pyarrow.parquet.write_table(table, path)


def __write_structure_collection(path: PathLike, collection: oc.StructureCollection):
    if len(collection) > 1:
        raise ValueError(
            "write_parquet currently only supports writing a single structure"
        )
    data: dict[str, np.ndarray] = {}
    for name, dataset in collection.items():
        if "particle" not in name:
            warn(f"write_parquet only supports writing particles, skipping {name}")
        assert isinstance(dataset, oc.Dataset)
        prefix = name.split("_")[0]
        particle_data = dataset.get_data("numpy")
        if not isinstance(particle_data, dict):
            particle_data = {dataset.columns[0]: particle_data}

        particle_data = {
            f"{prefix}_{cname}": cdata for cname, cdata in particle_data.items()
        }
        data.update(particle_data)
    table = pyarrow.table(data)
    pyarrow.parquet.write_table(table, path)
