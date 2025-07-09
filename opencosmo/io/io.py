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
    *files: str | Path | h5py.File | h5py.Group, **load_kwargs: bool
) -> oc.Dataset | collection.Collection:
    """
    Open a dataset or data collection from one or more opencosmo files.

    If you open a file with this function, you should generally close it
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
    *files: str or pathlib.Path
        The path(s) to the file(s) to open.

    **load_kwargs: bool
        True/False flags that can be used to only load certain datasets from
        the files. Check the documentation for the data type you are working
        with for available flags. Will be ignored if only one file is passed
        and the file only contains a single dataset.

    Returns
    -------
    dataset : oc.Dataset or oc.Collection
        The dataset or collection opened from the file.

    """
    # For now the only way to open multiple files is with a StructureCollection
    if len(files) == 1 and isinstance(files[0], list):
        return oc.open(*files[0])

    if len(files) > 1 and all(isinstance(f, (str, Path)) for f in files):
        paths = [resolve_path(path, FileExistance.MUST_EXIST) for path in files]
        headers = [read_header(p) for p in paths]
        if all(h.file.is_lightcone for h in headers):
            return oc.collection.open_lightcone(paths)

        return oc.open_linked_files(*paths, **load_kwargs)

    handles = get_file_handles(*files)
    if len(handles) == 1 and "data" in handles[0].keys():
        return open_single_dataset(handles[0])

    verify_files(handles)
    return collection.open_collection(handles, load_kwargs)


def expand_file(handle: h5py.File | h5py.Group):
    if "header" in handle.keys():
        return [handle]
    elif not all("header" in group.keys() for group in handle.values()):
        raise ValueError("The file is missing a header!")
    return list(handle.values())


def verify_files(handles: list[h5py.File | h5py.Group]):
    # if multiple files, one of the following must be true:
    # Files have the same set of datatypes
    first_headers = [read_header(g) for g in expand_file(handles[0])]
    expected_dtypes = set(h.file.data_type for h in first_headers)
    for handle in handles[1:]:
        headers = [read_header(g) for g in expand_file(handle)]
        dtypes = set(h.file.data_type for h in headers)
        if dtypes != expected_dtypes:
            raise ValueError("All files should have the same set of data types!")


def open_single_dataset(handle: h5py.File | h5py.Group):
    header = read_header(handle)
    try:
        tree = open_tree(handle, header.simulation.box_size, header.file.is_lightcone)
    except ValueError:
        tree = None
    if header.file.region is not None:
        sim_region = from_model(header.file.region)

    elif header.file.is_lightcone:
        sim_region = FullSkyRegion()

    else:
        p1 = (0, 0, 0)
        p2 = tuple(header.simulation.box_size for _ in range(3))
        sim_region = oc.make_box(p1, p2)

    index: ChunkedIndex
    handler = DatasetHandler(handle)
    if (comm := get_comm_world()) is not None:
        assert partition is not None
        part = partition(comm, len(handler), tree)
        index = part.idx
        sim_region = part.region if part.region is not None else sim_region
    else:
        index = ChunkedIndex.from_size(len(handler))

    builders, base_unit_transformations = u.get_default_unit_transformations(
        handle, header
    )
    state = dss.DatasetState(
        base_unit_transformations,
        builders,
        index,
        u.UnitConvention.COMOVING,
        sim_region,
        header,
    )

    dataset = oc.Dataset(
        handler,
        header,
        state,
        tree=tree,
    )

    if header.file.is_lightcone:
        return collection.Lightcone({"data": dataset})
    return dataset


pass


def get_file_handles(*files: str | Path | h5py.File | h5py.Group):
    handles = []
    for file in files:
        if not isinstance(file, h5py.File) and not isinstance(file, h5py.Group):
            path = resolve_path(file, FileExistance.MUST_EXIST)
            file_handle = h5py.File(path, "r")
            handles.append(file_handle)
            continue
        handles.append(file)
    return handles


def evaluate_load_conditions(
    groups: dict[str, h5py.Group], load_kwargs: dict[str, bool]
):
    """
    Datasets can define conditional loading via an addition group called "load/if".
    the "if" group can define parameters which must either be true or false for the
    given group to be loaded. These parameters can then be provided by the user to the
    "open" function. Parameters not specified by the user default to False.
    """
    output_groups = {}
    for name, group in groups.items():
        try:
            ifgroup = group["load/if"]
        except KeyError:
            output_groups[name] = group
            continue
        load = True
        for key, condition in ifgroup.attrs.items():
            load = load and (load_kwargs.get(key, False) == condition)
        if load:
            output_groups[name] = group
    return output_groups


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
        raise ValueError(
            "oc.read can not be used to read files with multiple datasets. Use oc.open"
        )
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
        base_unit_transformations,
        builders,
        index,
        u.UnitConvention.COMOVING,
        sim_box,
        header,
    )

    ds = oc.Dataset(handler, header, state, tree)

    if header.file.is_lightcone:
        return collection.Lightcone({"data": ds})
    return ds


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
