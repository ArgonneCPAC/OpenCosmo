from typing import Iterable, Optional
import h5py
from pathlib import Path
from opencosmo import dataset as ds
from opencosmo.header import OpenCosmoHeader, read_header
from opencosmo.collection import Collection, ParticleCollection, SimulationCollection
from opencosmo.collection.link import verify_links, get_links, LinkedCollection
from opencosmo.transformations import units as u
from opencosmo.handler import InMemoryHandler, OpenCosmoDataHandler, OutOfMemoryHandler
from opencosmo import io
from collections import defaultdict
from opencosmo.spatial import read_tree
import numpy as np

try:
    from mpi4py import MPI
    from opencosmo.handler import MPIHandler
except ImportError:
    MPI = None


class FileHandle:
    """
    Helper class used just for setup
    """

    def __init__(self, path: Path):
        self.handle = h5py.File(path, "r")
        self.header = read_header(self.handle)

def open_multi_dataset_file(file: h5py.File, datasets: Optional[Iterable[str]]) -> Collection:
    """
    Open a file with multiple datasets.
    """
    CollectionType = get_collection_type(file)
    return CollectionType.open(file, datasets)
    

def read_multi_dataset_file(
    file: h5py.File, datasets: Optional[Iterable[str]] = None
) -> Collection:
    """
    Read particle data from an HDF5 file.
    """
    CollectionType = get_collection_type(file)
    return CollectionType.read(file, datasets)

def open_linked(*files: Path):
    """
    Open a collection of files that are linked together, such as a
    properties file and a particle file.
    """
    file_handles = [FileHandle(file) for file in files]
    datasets = [io.open(file) for file in files]
    property_file_type, linked_files = verify_links(*[fh.header for fh in file_handles])
    property_handle = next(
        filter(lambda x: x.header.file.data_type == property_file_type, file_handles)
    ).handle
    links = get_links(property_handle)
    if not links:
        raise ValueError("No valid links found in files")

    output_datasets: dict[str, ds.Dataset] = {}
    for dataset in datasets:
        try:
            output_datasets.update(dataset.as_dict())
        except AttributeError:
            output_datasets[dataset.header.file.data_type] = dataset

    properties_file = output_datasets.pop(property_file_type)
    return LinkedCollection(
        properties_file.header, properties_file, output_datasets, links
    )
def get_collection_type(file: h5py.File) -> Collection:
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
