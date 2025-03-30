from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import h5py

from opencosmo import dataset as ds
from opencosmo.collection import Collection, ParticleCollection, SimulationCollection
from opencosmo.link.collection import LinkedCollection
from opencosmo.header import read_header


class FileHandle:
    """
    Helper class used just for setup
    """

    def __init__(self, path: Path):
        self.handle = h5py.File(path, "r")
        self.header = read_header(self.handle)


def open_multi_dataset_file(
    file: h5py.File, datasets: Optional[Iterable[str]]
) -> Collection | ds.Dataset:
    """
    Open a file with multiple datasets.
    """
    CollectionType = get_collection_type(file)
    return CollectionType.open(file, datasets)


def read_multi_dataset_file(
    file: h5py.File, datasets: Optional[Iterable[str]] = None
) -> Collection | ds.Dataset:
    """
    Read a file with multiple datasets.
    """
    CollectionType = get_collection_type(file)
    return CollectionType.read(file, datasets)


def get_collection_type(file: h5py.File) -> type[Collection]:
    """
    Determine the type of a single file containing multiple datasets. Currently
    we support multi_simulation, particle, and linked collections.

    multi_simulation == multiple simulations, same data types
    particle == single simulation, multiple particle species
    linked == A properties dataset, linked with other particle or profile datasets
    """
    datasets = [k for k in file.keys() if k != "header"]
    if len(datasets) == 0:
        raise ValueError("No datasets found in file.")

    if all("particle" in dataset for dataset in datasets) and "header" in file.keys():
        return ParticleCollection

    elif "header" not in file.keys():
        config_values = defaultdict(list)
        for dataset in datasets:
            try:
                filetype_data = dict(file[dataset]["header"]["file"].attrs)
                for key, value in filetype_data.items():
                    config_values[key].append(value)
            except KeyError:
                continue
        if all(len(set(v)) == 1 for v in config_values.values()):
            return SimulationCollection
        else:
            raise ValueError(
                "Unknown file type. "
                "It appears to have multiple datasets, but organized incorrectly"
            )
    elif len(list(filter(lambda x: x.endswith("properties"), datasets))) == 1:
        return LinkedCollection
    else:
        raise ValueError(
            "Unknown file type. "
            "It appears to have multiple datasets, but organized incorrectly"
        )
