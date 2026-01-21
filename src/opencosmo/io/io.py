from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional

import h5py

import opencosmo as oc
from opencosmo import collection
from opencosmo.dataset import state as dss
from opencosmo.dataset.handler import Hdf5Handler
from opencosmo.file import FileExistance, resolve_path
from opencosmo.index.build import empty, from_range
from opencosmo.io.file import (
    evaluate_load_conditions,
    get_file_type,
    make_all_targets,
)
from opencosmo.io.serial import allocate, write_columns, write_metadata
from opencosmo.mpi import get_comm_world
from opencosmo.spatial.builders import from_model
from opencosmo.spatial.region import FullSkyRegion
from opencosmo.spatial.tree import open_tree
from opencosmo.units import UnitConvention

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    from opencosmo.index import ChunkedIndex
    from opencosmo.io.file import OpenTarget

    from .protocols import Writeable

    mpiio: Optional[ModuleType]
    partition: Optional[Callable]

if get_comm_world() is not None:
    from opencosmo.dataset.mpi import partition
    from opencosmo.io import mpi as mpiio
else:
    mpiio = None
    partition = None

    """
    This module defines the main user-facing io functions: open and write

    open can take any number of file paths, and will always construct a single object 
    (either a dataset or a collection).

    write takes exactly one path and exactly one opencosmo dataset or collection

    open works in the following way:

    1. Read headers and get dataset names and types for all files passed
    2. If there is only a single dataset, simply open it as such
    3. If there are multiple datasets, user the headers to determine
       if the dataset are compatible (i.e. capabale of existing together in
       a collection)
    4. Open all datasets individually
    5. Call the merge functionality for the appropriate collection.
    """


class COLLECTION_TYPE(Enum):
    LIGHTCONE = 0
    STRUCTURE_COLLECTION = 1
    SIMULATION_COLLECTION = 2


def open(
    *files: str | Path | h5py.File | h5py.Group, **open_kwargs: bool
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

    When you have multiple files that can be combined into a collection,
    you can use the following.

    .. code-block:: python

        import opencosmo as oc
        ds = oc.open("haloproperties.hdf5", "haloparticles.hdf5")


    Parameters
    ----------
    *files: str or pathlib.Path
        The path(s) to the file(s) to open.

    **open_kwargs: bool
        True/False flags that can be used to only load certain datasets from
        the files. Check the documentation for the data type you are working
        with for available flags. Will be ignored if only one file is passed
        and the file only contains a single dataset.

    Returns
    -------
    dataset : oc.Dataset or oc.Collection
        The dataset or collection opened from the file.

    """
    if len(files) == 1 and isinstance(files[0], list):
        file_list = files[0]
    else:
        file_list = list(files)
    file_list.sort()

    try:
        handles = [h5py.File(f) for f in file_list]
    except TypeError:  # we have hdf5 groups
        handles = file_list

    try:
        targets = make_all_targets(handles)
    except KeyError:
        if len(handles) != 1:
            raise
        datasets = {name: oc.open(group) for name, group in handles[0].items()}
        return oc.SimulationCollection(datasets)

    targets = evaluate_load_conditions(targets, open_kwargs)
    file_types = list(map(get_file_type, handles))
    if len(targets) > 1:
        collection_type = collection.get_collection_type(targets, file_types)
        return collection_type.open(targets, **open_kwargs)

    else:
        return open_single_dataset(targets[0])

    # For now the only way to open multiple files is with a StructureCollection


def open_single_dataset(
    target: OpenTarget,
    metadata_group: Optional[str] = None,
    bypass_lightcone: bool = False,
    bypass_mpi: bool = False,
):
    header = target.header
    handle = target.group

    assert header is not None

    try:
        tree = open_tree(
            handle,
            header.with_units("scalefree").simulation["box_size"].value,
            header.file.is_lightcone,
        )
    except ValueError:
        tree = None

    if header.file.region is not None:
        sim_region = from_model(header.file.region)
    elif header.file.is_lightcone:
        sim_region = FullSkyRegion()
    else:
        p1 = (0, 0, 0)
        p2 = tuple(header.simulation["box_size"].value for _ in range(3))
        sim_region = oc.make_box(p1, p2)

    index: Optional[ChunkedIndex] = None
    handler = Hdf5Handler.from_group(handle["data"])

    if not bypass_mpi and (comm := get_comm_world()) is not None:
        assert partition is not None
        try:
            idx_data = handle["index"]
            part = partition(comm, len(handler), idx_data, tree)
            if part is None:
                index = empty()
            else:
                index = part.idx
                sim_region = part.region if part.region is not None else sim_region
        except KeyError:
            n_ranks = comm.Get_size()
            n_per = len(handler) // n_ranks
            chunk_boundaries = [i * n_per for i in range(n_ranks + 1)]
            chunk_boundaries[-1] = len(handler)
            rank = comm.Get_rank()
            index = from_range(chunk_boundaries[rank], chunk_boundaries[rank + 1])

    if metadata_group is not None:
        metadata_group = handle[metadata_group]

    elif "metadata" in handle.keys():
        metadata_group = handle["metadata"]

    state = dss.DatasetState.from_group(
        handle,
        header,
        UnitConvention.COMOVING,
        sim_region,
        index,
        metadata_group,
    )

    dataset = oc.Dataset(
        header,
        state,
        tree=tree,
    )
    if header.file.data_type == "healpix_map":
        return collection.HealpixMap(
            {"data": dataset},
            header.healpix_map["nside"],
            header.healpix_map["nside_lr"],
            header.healpix_map["ordering"],
            header.healpix_map["full_sky"],
            header.healpix_map["z_range"],
        )
    elif header.file.is_lightcone and not bypass_lightcone:
        return collection.Lightcone({"data": dataset}, header.lightcone["z_range"])

    return dataset


def write(path: Path, dataset: Writeable, overwrite=False, **schema_kwargs) -> None:
    """
    Write a dataset or collection to the file at the sepecified path.

    Parameters
    ----------
    file : str or pathlib.Path
        The path to the file to write to.
    dataset : oc.Dataset
        The dataset to write.
    overwrite : bool, default = False
        If the file already exists, overwrite it


    Raises
    ------
    FileExistsError
        If the file at the specified path already exists and overwrite is False
    FileNotFoundError
        If the parent folder of the ouput file does not exist
    """

    existance_requirement = FileExistance.MUST_NOT_EXIST
    if overwrite:
        existance_requirement = FileExistance.EITHER

    path = resolve_path(path, existance_requirement)

    schema = dataset.make_schema(**schema_kwargs)

    if mpiio is not None:
        return mpiio.write_parallel(path, schema)

    file = h5py.File(path, "w")
    allocate(file, schema)
    write_metadata(file, schema)
    write_columns(file, schema)
