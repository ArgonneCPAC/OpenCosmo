from __future__ import annotations

from typing import Iterable, Optional, Protocol

try:
    from mpi4py import MPI

    from opencosmo.handler import MPIHandler
except ImportError:
    MPI = None  # type: ignore


import h5py
import numpy as np

import opencosmo as oc
from opencosmo.handler import InMemoryHandler, OpenCosmoDataHandler, OutOfMemoryHandler
from opencosmo.header import OpenCosmoHeader, read_header
from opencosmo.spatial import read_tree
from opencosmo.transformations import units as u


class Collection(Protocol):
    """
    Collections represent a group of datasets that are related in some way. They
    support higher-level operations that are applied across all datasets in the
    collection, sometimes in a non-obvious way.

    This protocl defines methods a collection must implement. Note that 
    the "open" and "read" methods are used in the case an entire collection
    is located within a single file. Multi-file collections are handled
    in the collection.io module. Most complexity is hidden from the user
    who simply calls "oc.read" and "oc.open" to get a collection. The io 
    module also does sanity checking to ensure files are structurally valid, 
    so we do not have to do it here.
    """
    @classmethod
    def open(
        cls, file: h5py.File, datasets_to_get: Optional[Iterable[str]] = None
    ) -> Collection | oc.Dataset: ...

    @classmethod
    def read(
        cls, file: h5py.File, datasets_to_get: Optional[Iterable[str]]
    ) -> Collection: ...

    def write(self, file: h5py.File): ...

    def as_dict(self) -> dict[str, oc.Dataset]: ...

    def __enter__(self): ...

    def __exit__(self, *exc_details): ...


def write_with_common_header(
    collection: Collection, header: OpenCosmoHeader, file: h5py.File
):
    """
    Write a collection to an HDF5 file when all datasets share
    a common header.
    """
    # figure out if we have unique headers

    header.write(file)
    for key, dataset in collection.as_dict().items():
        dataset.write(file, key, with_header=False)


def write_with_unique_headers(collection: Collection, file: h5py.File):
    """
    Write the collection to an HDF5 file when each dattaset
    has its own header.
    """
    # figure out if we have unique headers

    for key, dataset in collection.as_dict().items():
        dataset.write(file, key)


def verify_datasets_exist(file: h5py.File, datasets: Iterable[str]):
    """
    Verify a set of datasets exist in a given file.
    """
    if not set(datasets).issubset(set(file.keys())):
        raise ValueError(f"Some of {', '.join(datasets)} not found in file.")


class ParticleCollection(dict):
    def __init__(self, header: OpenCosmoHeader, datasets: dict[str, oc.Dataset]):
        """
        Represents a collection of datasets for different particle species
        from a single simulation. All should share the same header.
        """
        self.__header = header
        self.update(datasets)

    @property
    def header(self):
        return self.__header

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        for dataset in self.values():
            try:
                dataset.close()
            except ValueError:
                continue

    @classmethod
    def open(
        cls, file: h5py.File, datasets_to_get: Optional[Iterable[str]] = None
    ) -> ParticleCollection | oc.Dataset:
        if datasets_to_get is not None:
            verify_datasets_exist(file, datasets_to_get)
            names = datasets_to_get
        else:
            names = list(filter(lambda x: x != "header", file.keys()))

        header = read_header(file)
        datasets = {name: open_single_dataset(file, name, header) for name in names}
        if not datasets:
            raise ValueError("No datasets found in file.")

        elif len(datasets) == 1:
            return next(iter(datasets.values()))
        return cls(header, datasets)

    @classmethod
    def read(
        cls, file: h5py.File, datasets_to_get: Optional[Iterable[str]] = None
    ) -> ParticleCollection:
        if datasets_to_get is not None:
            verify_datasets_exist(file, datasets_to_get)
            names = datasets_to_get
        else:
            names = list(filter(lambda x: x != "header", file.keys()))

        header = read_header(file)
        datasets = {name: read_single_dataset(file, name, header) for name in names}

        if not datasets:
            raise ValueError("No datasets found in file.")

        elif len(datasets) == 1:
            return next(iter(datasets.values()))

        return cls(header, datasets)

    particle_types = dict.keys
    particles = dict.values

    def write(self, file: h5py.File):
        return write_with_common_header(self, self.__header, file)

    def as_dict(self) -> dict[str, oc.Dataset]:
        return self


class SimulationCollection(dict):
    """
    A collection of datasets of the same type from different
    simulations. In general this exposes the exact same API
    as the individual datasets, but maps the results across
    all of them.
    """

    def __init__(self, dtype: str, datasets: dict[str, oc.Dataset]):
        self.dtype = dtype
        self.update(datasets)

    def as_dict(self) -> dict[str, oc.Dataset]:
        return self

    @classmethod
    def open(
        cls, file: h5py.File, datasets_to_get: Optional[Iterable[str]] = None
    ) -> SimulationCollection:
        if datasets_to_get is not None:
            verify_datasets_exist(file, datasets_to_get)
            names = datasets_to_get
        else:
            names = list(filter(lambda x: x != "header", file.keys()))
        datasets = {name: open_single_dataset(file, name) for name in names}
        dtype = next(iter(datasets.values())).header.file.data_type
        return cls(dtype, datasets)

    @classmethod
    def read(
        cls, file: h5py.File, datasets_to_get: Optional[Iterable[str]] = None
    ) -> SimulationCollection:
        if datasets_to_get is not None:
            verify_datasets_exist(file, datasets_to_get)
            names = datasets_to_get
        else:
            names = list(filter(lambda x: x != "header", file.keys()))

        datasets = {name: read_single_dataset(file, name) for name in names}
        dtype = next(iter(datasets.values())).header.file.data_type
        return cls(dtype, datasets)

    datasets = dict.values

    def write(self, h5file: h5py.File):
        return write_with_unique_headers(self, h5file)

    def __map(self, method, *args, **kwargs):
        """
        This type of collection will only ever be constructed if all the underlying
        datasets have the same data type, so it is always safe to map operations
        across all of them.
        """
        output = {k: getattr(v, method)(*args, **kwargs) for k, v in self.items()}
        return SimulationCollection(self.dtype, output)

    def __getattr__(self, name):
        # check if the method exists on the first dataset
        if hasattr(next(iter(self.values())), name):
            return lambda *args, **kwargs: self.__map(name, *args, **kwargs)
        else:
            raise AttributeError(f"Attribute {name} not found on {self.dtype} dataset")


def open_single_dataset(
    file: h5py.File, dataset_key: str, header: Optional[OpenCosmoHeader] = None
) -> oc.Dataset:
    """
    Open a single dataset in a file with multiple datasets.
    """
    if dataset_key not in file.keys():
        raise ValueError(f"No group named '{dataset_key}' found in file.")

    if header is None:
        header = read_header(file[dataset_key])

    tree = read_tree(file[dataset_key], header)
    handler: OpenCosmoDataHandler
    if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
        handler = MPIHandler(
            file, tree=tree, comm=MPI.COMM_WORLD, group_name=dataset_key
        )
    else:
        handler = OutOfMemoryHandler(file, tree=tree, group_name=dataset_key)

    builders, base_unit_transformations = u.get_default_unit_transformations(
        file[dataset_key], header
    )
    mask = np.arange(len(handler))
    return oc.Dataset(handler, header, builders, base_unit_transformations, mask)


def read_single_dataset(
    file: h5py.File, dataset_key: str, header: Optional[OpenCosmoHeader] = None
):
    """
    Read a single dataset from a multi-dataset file
    """
    if dataset_key not in file.keys():
        raise ValueError(f"No group named '{dataset_key}' found in file.")

    if header is None:
        header = read_header(file[dataset_key])

    tree = read_tree(file[dataset_key], header)
    handler = InMemoryHandler(file, tree, dataset_key)
    builders, base_unit_transformations = u.get_default_unit_transformations(
        file[dataset_key], header
    )
    mask = np.arange(len(handler))
    return oc.Dataset(handler, header, builders, base_unit_transformations, mask)
