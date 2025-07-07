from pathlib import Path
from typing import Type

import h5py

import opencosmo as oc
from opencosmo import dataset as ds
from opencosmo.collection.lightcone import Lightcone
from opencosmo.collection.protocols import Collection
from opencosmo.collection.simulation import SimulationCollection
from opencosmo.collection.structure import StructureCollection


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
    dtypes = set(dataset for dataset in datasets.values())
    if len(dtypes) != 1:
        raise ValueError("All datasets must be of the same type.")
    return SimulationCollection(datasets)


def open_collection(
    handles: list[h5py.File | h5py.Group], load_kwargs: dict[str, bool]
) -> Collection | ds.Dataset:
    """
    Open a file with multiple datasets.
    """
    CollectionType = get_collection_type(handles)
    return CollectionType.open(handles, load_kwargs)


def get_collection_type(handles: list[h5py.File | h5py.Group]) -> Type[Collection]:
    """
    Determine the type of a single file containing multiple datasets. Currently
    we support multi_simulation, particle, and linked collections.

    multi_simulation == multiple simulations, same data types
    particle == single simulation, multiple particle species
    linked == A properties dataset, linked with other particle or profile datasets
    """
    if len(handles) > 1:
        return SimulationCollection

    handle = handles[0]

    datasets = [k for k in handle.keys() if k != "header"]
    if len(datasets) == 0:
        raise ValueError("No datasets found in file.")

    if "header" not in handle.keys():
        if all(group["header/file"].attrs["is_lightcone"] for group in handle.values()):
            return Lightcone
        return SimulationCollection
    elif len(list(filter(lambda x: x.endswith("properties"), datasets))) >= 1:
        return StructureCollection
    else:
        raise ValueError(
            "Unknown file type. "
            "It appears to have multiple datasets, but organized incorrectly"
        )
