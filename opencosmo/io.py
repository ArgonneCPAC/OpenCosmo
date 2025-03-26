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
from opencosmo import collection
from opencosmo.file import FileExistance, file_reader, file_writer, resolve_path
from opencosmo.handler import InMemoryHandler, OpenCosmoDataHandler, OutOfMemoryHandler
from opencosmo.header import read_header
from opencosmo.spatial import read_tree
from opencosmo.transformations import units as u


def open(
    file: str | Path, datasets: Optional[str | Iterable[str]] = None
) -> oc.Dataset | collection.Collection:
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
        if not isinstance(datasets, str):
            return collection.open_multi_dataset_file(file_handle, datasets)
        try:
            group = file_handle[datasets]
        except KeyError:
            raise ValueError(f"Dataset {datasets} not found in file {file}")
    else:
        group = file_handle

    header = read_header(file_handle)
    tree = read_tree(file_handle, header)
    if datasets is not None and not isinstance(datasets, str):
        raise ValueError("Asked for multiple datasets, but file has only one")

    handler: OpenCosmoDataHandler
    if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
        handler = MPIHandler(
            file_handle, group_name=datasets, tree=tree, comm=MPI.COMM_WORLD
        )
    else:
        handler = OutOfMemoryHandler(file_handle, group_name=datasets, tree=tree)

    builders, base_unit_transformations = u.get_default_unit_transformations(
        group, header
    )
    mask = np.arange(len(handler))

    dataset = oc.Dataset(handler, header, builders, base_unit_transformations, mask)
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
    tree = read_tree(file, header)
    handler = InMemoryHandler(file, tree, group_name=datasets)
    mask = np.arange(len(handler))
    builders, base_unit_transformations = u.get_default_unit_transformations(
        group, header
    )

    return oc.Dataset(handler, header, builders, base_unit_transformations, mask)


@file_writer
def write(file: h5py.File, dataset: oc.Dataset | collection.Collection) -> None:
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
