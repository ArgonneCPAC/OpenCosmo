from __future__ import annotations

from pathlib import Path

import h5py

try:
    from mpi4py import MPI

    from opencosmo.handler import MPIHandler
except ImportError:
    MPI = None  # type: ignore
from typing import Iterable, Optional

import numpy as np

import opencosmo as oc
from opencosmo.dataset.collection import DataCollection, get_collection_type
from opencosmo.file import FileExistance, file_reader, file_writer, resolve_path
from opencosmo.handler import InMemoryHandler, OpenCosmoDataHandler, OutOfMemoryHandler
from opencosmo.header import OpenCosmoHeader, read_header
from opencosmo.spatial import read_tree
from opencosmo.transformations import units as u


def open(
    file: str | Path, datasets: Optional[str | Iterable[str]] = None
) -> oc.Dataset | DataCollection:
    """
    Open a dataset from a file without reading the data into memory.

    The object returned by this function will only read data from the file
    when it is actually needed. This is useful if the file is very large
    and you only need to access a small part of it.

    If you open a file with this dataset, you should generally close it
    when you're done

    .. code-block:: python

        import opencosmo as oc
        ds = oc.open("path/to/file.hdf5")
        # do work
        ds.close()

    Alternatively you can use a context manager, which will close the file
    automatically when you are done with it.

    .. code-block:: python

        import opencosmo as oc
        with oc.open("path/to/file.hdf5") as ds:
            # do work

    Parameters
    ----------
    file : str or pathlib.Path
        The path to the file to open.
    """
    path = resolve_path(file, FileExistance.MUST_EXIST)
    file_handle = h5py.File(path, "r")
    if "data" not in file_handle:
        return open_multi_dataset_file(file_handle)

    header = read_header(file_handle)
    tree = read_tree(file_handle, header)

    handler: OpenCosmoDataHandler
    if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
        handler = MPIHandler(file_handle, tree=tree, comm=MPI.COMM_WORLD)
    else:
        handler = OutOfMemoryHandler(file_handle, tree=tree)

    builders, base_unit_transformations = u.get_default_unit_transformations(
        file_handle, header
    )
    mask = np.ones(len(handler), dtype=bool)

    dataset = oc.Dataset(handler, header, builders, base_unit_transformations, mask)
    return dataset


@file_reader
def read(
    file: h5py.File, datasets: Optional[str | Iterable[str]] = None
) -> oc.Dataset | DataCollection:
    """
    Read a dataset from a file into memory.

    You should use this function if the data are small enough that having
    a copy of it (or a few copies of it) in memory is not a problem. For
    larger datasets, use :py:func:`opencosmo.open`.

    Parameters
    ----------
    file : str or pathlib.Path
        The path to the file to read.
    Returns
    -------
    dataset : oc.Dataset
        The dataset read from the file.

    """
    if "data" not in file:
        if isinstance(datasets, str):
            return read_multi_dataset_file(file, [datasets])
        else:
            return read_multi_dataset_file(file, datasets)

    header = read_header(file)
    tree = read_tree(file, header)
    handler = InMemoryHandler(file, tree)
    mask = np.ones(len(handler), dtype=bool)
    builders, base_unit_transformations = u.get_default_unit_transformations(
        file, header
    )

    return oc.Dataset(handler, header, builders, base_unit_transformations, mask)


@file_writer
def write(file: h5py.File, dataset: oc.Dataset | oc.DataCollection) -> None:
    """
    Write a dataset to a file.

    Parameters
    ----------
    file : str or pathlib.Path
        The path to the file to write to.
    dataset : oc.Dataset
        The dataset to write.

    """
    dataset.write(file)


def open_multi_dataset_file(file: h5py.File) -> DataCollection:
    """
    Open a file with multiple datasets.
    """
    CollectionType = get_collection_type(file)
    try:
        header = read_header(file)
    except KeyError:
        header = None

    datasets = [k for k in file.keys() if k != "header"]
    if len(datasets) == 0:
        raise ValueError("No datasets found in file.")
    collection = CollectionType(header=header)
    for dataset_name in datasets:
        collection[dataset_name] = open_single_dataset(file, dataset_name, header)

    if len(collection) == 1:
        return list(collection.values())[0]
    return collection


def open_single_dataset(
    file: h5py.File, dataset_key: str, header: Optional[OpenCosmoHeader] = None
) -> oc.Dataset:
    """
    Open a file with a single dataset.
    """
    if dataset_key not in file.keys():
        raise ValueError(f"No group named '{dataset_key}' found in file.")

    if header is None:
        header = read_header(file[dataset_key])

    tree = read_tree(file[dataset_key], header)
    handler: OpenCosmoDataHandler
    if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
        handler = MPIHandler(file, tree=tree, comm=MPI.COMM_WORLD, group=dataset_key)
    else:
        handler = OutOfMemoryHandler(file, tree=tree, group=dataset_key)

    builders, base_unit_transformations = u.get_default_unit_transformations(
        file[dataset_key], header
    )
    mask = np.ones(len(handler), dtype=bool)
    return oc.Dataset(handler, header, builders, base_unit_transformations, mask)


def read_multi_dataset_file(
    file: h5py.File, datasets: Optional[Iterable[str]] = None
) -> DataCollection:
    """
    Read particle data from an HDF5 file.
    """
    CollectionType = get_collection_type(file)
    try:
        header = read_header(file)
    except KeyError:
        header = None

    datasets_in_file = {k for k in file.keys() if k != "header"}
    if len(datasets_in_file) == 0:
        raise ValueError("No datasets found in file.")

    if datasets is not None:
        requested_datasets = set(datasets)
        missing_datasets = set(datasets) - requested_datasets
        if missing_datasets:
            raise ValueError(f"Datasets {missing_datasets} not found in file.")
        datasets_in_file = set(requested_datasets)

    collection = CollectionType(header=header)
    for dataset_name in datasets_in_file:
        collection[dataset_name] = read_single_dataset(file, dataset_name, header)

    if len(collection) == 1:
        return list(collection.values())[0]
    return collection


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
    mask = np.ones(len(handler), dtype=bool)
    return oc.Dataset(handler, header, builders, base_unit_transformations, mask)
