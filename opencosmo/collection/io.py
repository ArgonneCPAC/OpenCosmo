from pathlib import Path
from typing import Iterable, Optional

import h5py

import opencosmo as oc
from opencosmo import dataset as ds
from opencosmo.collection import Collection, SimulationCollection
from opencosmo.link.collection import StructureCollection


def open_simulation_files(**paths: Path) -> SimulationCollection:
    """
    Open multiple files and return a simulation collection. The data
    type of every file must be the same.

    Parameters
    ----------
    paths : str or Path
        The paths to the files to open.

    Returns
    -------
    SimulationCollection

    """
    datasets: dict[str, oc.Dataset] = {}
    for key, path in paths.items():
        dataset = oc.open(path)
        if not isinstance(dataset, oc.Dataset):
            raise ValueError("All datasets must be of the same type.")
    dtypes = set(dataset.header.file.data_type for dataset in datasets.values())
    if len(dtypes) != 1:
        raise ValueError("All datasets must be of the same type.")
    return SimulationCollection(datasets)


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

    if "header" not in file.keys():
        return SimulationCollection
    elif len(list(filter(lambda x: x.endswith("properties"), datasets))) >= 1:
        return StructureCollection
    else:
        raise ValueError(
            "Unknown file type. "
            "It appears to have multiple datasets, but organized incorrectly"
        )
