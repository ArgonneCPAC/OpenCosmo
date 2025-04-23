from __future__ import annotations

from pathlib import Path

import h5py

try:
    from mpi4py import MPI

    from opencosmo.handler import MPIHandler
    from opencosmo.handler.mpi import partition
except ImportError:
    MPI = None  # type: ignore
from typing import Iterable, Optional

import opencosmo as oc
from opencosmo import collection
from opencosmo.dataset.index import ChunkedIndex, DataIndex
from opencosmo.file import FileExistance, file_reader, file_writer, resolve_path
from opencosmo.handler import InMemoryHandler, OpenCosmoDataHandler, OutOfMemoryHandler
from opencosmo.header import read_header
from opencosmo.transformations import units as u


def open(
    file: str | Path | h5py.File | h5py.Group,
    datasets: Optional[str | Iterable[str]] = None,
) -> oc.Dataset | collection.Collection:
    """
    Open a dataset or data collection from a file without reading the data into memory.

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
    datasets : str or list[str], optional
        If the file has multiple datasets, the name of the dataset(s) to open.
        All other datasets will be ignored. If not provided, will open all
        datasets

    Returns
    -------
    dataset : oc.Dataset or oc.Collection
        The dataset or collection opened from the file.

    """
    if not isinstance(file, h5py.File) and not isinstance(file, h5py.Group):
        path = resolve_path(file, FileExistance.MUST_EXIST)
        file_handle = h5py.File(path, "r")
    else:
        file_handle = file
    if "data" not in file_handle:
        if not isinstance(datasets, str):
            return collection.open_multi_dataset_file(file_handle, datasets)
        try:
            group = file_handle[datasets]
        except KeyError:
            raise ValueError(f"Dataset {datasets} not found in file {file}")
    else:
        group = file_handle

    header = read_header(file_handle)
    # tree = read_tree(file_handle, header)
    tree = None
    if datasets is not None and not isinstance(datasets, str):
        raise ValueError("Asked for multiple datasets, but file has only one")

    handler: OpenCosmoDataHandler
    index: DataIndex
    if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
        handler = MPIHandler(
            file_handle, group_name=datasets, tree=tree, comm=MPI.COMM_WORLD
        )
        start, size = partition(MPI.COMM_WORLD, len(handler))
        index = ChunkedIndex.single_chunk(start, size)
    else:
        handler = OutOfMemoryHandler(file_handle, group_name=datasets, tree=tree)
        index = ChunkedIndex.from_size(len(handler))
    builders, base_unit_transformations = u.get_default_unit_transformations(
        group, header
    )

    dataset = oc.Dataset(handler, header, builders, base_unit_transformations, index)
    return dataset


@file_reader
def read(
    file: h5py.File, datasets: Optional[str | Iterable[str]] = None
) -> oc.Dataset | collection.Collection:
    """
    Read a dataset from a file into memory.

    You should use this function if the data are small enough that having
    a copy of it (or a few copies of it) in memory is not a problem. For
    larger datasets, use :py:func:`opencosmo.open`.

    Note that some dataset types cannot be read, due to complexities with
    how the data is handled. Using :py:func:`opencosmo.open` is recommended
    for most use cases.

    Parameters
    ----------
    file : str or pathlib.Path
        The path to the file to read.
    datasets : str or list[str], optional
        If the file has multiple datasets, the name of the dataset(s) to read.
        All other datasets will be ignored. If not provided, will read all
        datasets

    Returns
    -------
    dataset : oc.Dataset or oc.Collection
        The dataset or collection read from the file.

    """

    if "data" not in file:
        if not isinstance(datasets, str):
            return collection.read_multi_dataset_file(file, datasets)
        try:
            group = file[datasets]
        except KeyError:
            raise ValueError(f"Dataset {datasets} not found in file {file}")
    else:
        group = file

    if datasets is not None and not isinstance(datasets, str):
        raise ValueError("Asked for multiple datasets, but file has only one")
    header = read_header(file)
    # tree = read_tree(file, header)
    tree = None
    handler = InMemoryHandler(file, tree, group_name=datasets)
    index = ChunkedIndex.from_size(len(handler))
    builders, base_unit_transformations = u.get_default_unit_transformations(
        group, header
    )

    return oc.Dataset(handler, header, builders, base_unit_transformations, index)


@file_writer
def write(file: h5py.File, dataset: oc.Dataset | collection.Collection) -> None:
    """
    Write a dataset or collection to the file at the sepecified path.

    Parameters
    ----------
    file : str or pathlib.Path
        The path to the file to write to.
    dataset : oc.Dataset
        The dataset to write.

    Raises
    ------
    FileExistsError
        If the file at the specified path already exists
    FileNotFoundError
        If the parent folder of the ouput file does not exist
    """
    dataset.write(file)
