from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, Optional

import h5py
from deprecated import deprecated  # type: ignore

import opencosmo as oc
from opencosmo import collection
from opencosmo.dataset import state as dss
from opencosmo.dataset.handler import DatasetHandler
from opencosmo.file import FileExistance, file_reader, file_writer, resolve_path
from opencosmo.header import read_header
from opencosmo.index import ChunkedIndex
from opencosmo.mpi import get_comm_world
from opencosmo.spatial.builders import from_model
from opencosmo.spatial.region import FullSkyRegion
from opencosmo.spatial.tree import open_tree, read_tree
from opencosmo.transformations import units as u

from .protocols import Writeable
from .schemas import FileSchema

mpiio: Optional[ModuleType]
partition: Optional[Callable]

if get_comm_world() is not None:
    from opencosmo.dataset.mpi import partition
    from opencosmo.io import mpi as mpiio
else:
    mpiio = None
    partition = None


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
    try:
        tree = open_tree(
            file_handle, header.simulation.box_size, header.file.is_lightcone
        )
    except ValueError:
        tree = None
    if datasets is not None and not isinstance(datasets, str):
        raise ValueError("Asked for multiple datasets, but file has only one")

    if header.file.region is not None:
        sim_region = from_model(header.file.region)

    elif header.file.is_lightcone:
        sim_region = FullSkyRegion()

    else:
        p1 = (0, 0, 0)
        p2 = tuple(header.simulation.box_size for _ in range(3))
        sim_region = oc.make_box(p1, p2)

    index: ChunkedIndex
    handler = DatasetHandler(file_handle, group_name=datasets)
    if (comm := get_comm_world()) is not None:
        assert partition is not None
        part = partition(comm, len(handler), tree)
        index = part.idx
        sim_region = part.region if part.region is not None else sim_region
    else:
        index = ChunkedIndex.from_size(len(handler))

    builders, base_unit_transformations = u.get_default_unit_transformations(
        group, header
    )
    state = dss.DatasetState(
        base_unit_transformations,
        builders,
        index,
        u.UnitConvention.COMOVING,
        sim_region,
    )

    dataset = oc.Dataset(
        handler,
        header,
        state,
        tree=tree,
    )
    return dataset


@deprecated(
    version="0.7",
    reason="oc.read is deprecated and will be removed in version 1.0. "
    "Please use oc.open instead",
)
@file_reader
def read(
    file: h5py.File, datasets: Optional[str | Iterable[str]] = None
) -> oc.Dataset | collection.Collection:
    """
    **WARNING: THIS METHOD IS DEPRECATED AND WILL BE REMOVED IN A FUTURE
    VERSION. USE** :py:meth:`opencosmo.open`


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
    try:
        tree = read_tree(file, header.simulation.box_size)
    except ValueError:
        tree = None
    p1 = (0, 0, 0)
    p2 = tuple(header.simulation.box_size for _ in range(3))
    sim_box = oc.make_box(p1, p2)

    path = file.filename
    file = h5py.File(path, driver="core")

    handler = DatasetHandler(file, group_name=datasets)
    index = ChunkedIndex.from_size(len(handler))
    builders, base_unit_transformations = u.get_default_unit_transformations(
        group, header
    )
    state = dss.DatasetState(
        base_unit_transformations, builders, index, u.UnitConvention.COMOVING, sim_box
    )

    return oc.Dataset(handler, header, state, tree)


@file_writer
def write(path: Path, dataset: Writeable) -> None:
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

    schema = FileSchema()
    dataset_schema = dataset.make_schema()
    schema.add_child(dataset_schema, "root")

    if mpiio is not None:
        return mpiio.write_parallel(path, schema)

    file = h5py.File(path, "w")
    schema.allocate(file)

    writer = schema.into_writer()

    writer.write(file)
