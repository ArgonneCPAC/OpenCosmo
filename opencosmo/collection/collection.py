from __future__ import annotations

from typing import Iterable, Mapping, Optional, Protocol

try:
    from mpi4py import MPI

    from opencosmo.handler import MPIHandler
except ImportError:
    MPI = None  # type: ignore


import h5py

import opencosmo as oc
from opencosmo.dataset.index import ChunkedIndex
from opencosmo.dataset.mask import Mask
from opencosmo.handler import InMemoryHandler, OpenCosmoDataHandler, OutOfMemoryHandler
from opencosmo.header import OpenCosmoHeader, read_header
from opencosmo.link import StructureCollection
from opencosmo.transformations import units as u


class Collection(Protocol):
    """
    Collections represent a group of datasets that are related in some way. They
    support higher-level operations that are applied across all datasets in the
    collection, sometimes in a non-obvious way.

    This protocol defines methods a collection must implement. Most notably they
    must include  __getitem__, keys, values and __items__, which allows
    a collection to behave like a read-only dictionary.


    Note that the "open" and "read" methods are used in the case an entire collection
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

    def __getitem__(self, key: str) -> oc.Dataset: ...
    def keys(self) -> Iterable[str]: ...
    def values(self) -> Iterable[oc.Dataset]: ...
    def items(self) -> Iterable[tuple[str, oc.Dataset]]: ...
    def __enter__(self): ...
    def __exit__(self, *exc_details): ...
    def filter(self, *masks: Mask) -> Collection: ...
    def select(self, *args, **kwargs) -> Collection: ...
    def with_units(self, convention: str) -> Collection: ...
    def take(self, *args, **kwargs) -> Collection: ...


def write_with_common_header(
    collection: Collection, header: OpenCosmoHeader, file: h5py.File
):
    """
    Write a collection to an HDF5 file when all datasets share
    a common header.
    """
    # figure out if we have unique headers

    header.write(file)
    keys = list(collection.keys())
    keys.sort()
    for key in keys:
        group = file.create_group(key)
        collection[key].write(group, key, with_header=False)


def write_with_unique_headers(collection: Collection, file: h5py.File):
    """
    Write the collection to an HDF5 file when each dattaset
    has its own header.
    """
    # figure out if we have unique headers

    keys = list(collection.keys())
    keys.sort()
    for key in keys:
        group = file.create_group(key)
        collection[key].write(group)


def verify_datasets_exist(file: h5py.File, datasets: Iterable[str]):
    """
    Verify a set of datasets exist in a given file.
    """
    if not set(datasets).issubset(set(file.keys())):
        raise ValueError(f"Some of {', '.join(datasets)} not found in file.")


class SimulationCollection(dict):
    """
    A collection of datasets of the same type from different
    simulations. In general this exposes the exact same API
    as the individual datasets, but maps the results across
    all of them.
    """

    def __init__(self, datasets: Mapping[str, oc.Dataset | Collection]):
        self.update(datasets)

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        for dataset in self.values():
            try:
                dataset.close()
            except ValueError:
                continue

    def __repr__(self):
        n_collections = sum(
            1
            for v in self.values()
            if isinstance(v, (SimulationCollection, StructureCollection))
        )
        n_datasets = sum(1 for v in self.values() if isinstance(v, oc.Dataset))
        return (
            f"SimulationCollection({n_collections} collections, {n_datasets} datasets)"
        )

    @classmethod
    def open(
        cls, file: h5py.File, datasets_to_get: Optional[Iterable[str]] = None
    ) -> SimulationCollection:
        if datasets_to_get is not None:
            verify_datasets_exist(file, datasets_to_get)
            names = datasets_to_get
        else:
            names = list(filter(lambda x: x != "header", file.keys()))
        datasets = {name: oc.open(file[name]) for name in names}
        return cls(datasets)

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
        return cls(datasets)

    def write(self, h5file: h5py.File):
        return write_with_unique_headers(self, h5file)

    def __map(self, method, *args, **kwargs):
        """
        This type of collection will only ever be constructed if all the underlying
        datasets have the same data type, so it is always safe to map operations
        across all of them.
        """
        output = {k: getattr(v, method)(*args, **kwargs) for k, v in self.items()}
        return SimulationCollection(output)

    def filter(self, *masks: Mask, **kwargs) -> SimulationCollection:
        """
        Filter the datasets in the collection. This method behaves
        exactly like :meth:`opencosmo.Dataset.filter` or
        :meth:`opencosmo.StructureCollection.filter`, but
        it applies the filter to all the datasets or collections
        within this collection. The result is a new collection.

        Parameters
        ----------
        filters:
            The filters constructed with :func:`opencosmo.col`

        Returns
        -------
        SimulationCollection
            A new collection with the same datasets, but only the
            particles that pass the filter.
        """
        return self.__map("filter", *masks, **kwargs)

    def select(self, *args, **kwargs) -> SimulationCollection:
        """
        Select a subset of the datasets in the collection. This method
        calls the underlying method in :class:`opencosmo.Dataset`, or
        :class:`opencosmo.Collection` depending on the context. As such
        its behavior and arguments can vary depending on what this collection
        contains.

        Parameters
        ----------
        args:
            The arguments to pass to the select method. This is
            usually a list of column names to select.
        kwargs:
            The keyword arguments to pass to the select method.
            This is usually a dictionary of column names to select.

        """
        return self.__map("select", *args, **kwargs)

    def with_units(self, convention: str) -> SimulationCollection:
        """
        Transform all datasets or collections to use the given unit convention. This
        method behaves exactly like :meth:`opencosmo.Dataset.with_units`.

        Parameters
        ----------
        convention: str
            The unit convention to use. One of "unitless",
            "scalefree", "comoving", or "physical".

        """
        return self.__map("with_units", convention)

    def take(self, *args, **kwargs) -> SimulationCollection:
        """
        Take a subest of rows from all datasets or collections in this collection.
        This method will delegate to the underlying method in
        :class:`opencosmo.Dataset`, or :class:`opencosmo.Collection` depending on  the
        context. As such, behaviormay vary depending on what this collection contains.
        """

        return self.__map("take", *args, **kwargs)


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

    # tree = read_tree(file[dataset_key], header)
    tree = None
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
    index = ChunkedIndex.from_size(len(handler))
    return oc.Dataset(handler, header, builders, base_unit_transformations, index)


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

    # tree = read_tree(file[dataset_key], header)
    tree = None

    handler = InMemoryHandler(file, tree, dataset_key)
    builders, base_unit_transformations = u.get_default_unit_transformations(
        file[dataset_key], header
    )
    index = ChunkedIndex.from_size(len(handler))
    return oc.Dataset(handler, header, builders, base_unit_transformations, index)
