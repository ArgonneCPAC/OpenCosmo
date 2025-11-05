from __future__ import annotations

from enum import Enum
from functools import reduce
from typing import TYPE_CHECKING, Callable, Iterable, Optional

import h5py
from deprecated import deprecated  # type: ignore

import opencosmo as oc
from opencosmo import collection
from opencosmo.dataset import state as dss
from opencosmo.dataset.handler import Hdf5Handler
from opencosmo.file import FileExistance, file_reader, resolve_path
from opencosmo.header import read_header
from opencosmo.index import ChunkedIndex
from opencosmo.mpi import get_comm_world
from opencosmo.spatial.builders import from_model
from opencosmo.spatial.region import FullSkyRegion
from opencosmo.spatial.tree import open_tree, read_tree
from opencosmo.units import UnitConvention
from opencosmo.units.get import get_unit_applicators_hdf5
from opencosmo.units.handler import make_unit_handler

from .schemas import FileSchema

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    from opencosmo.header import OpenCosmoHeader

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


class FILE_TYPE(Enum):
    HALO_PROPERTIES = 0
    HALO_PARTICLES = 1
    GALAXY_PROPERTIES = 2
    GALAXY_PARTICLES = 3
    SOD_BINS = 4
    LIGHTCONE = 5
    STRUCTURE_COLLECTION = 6
    SIMULATION_COLLECTION = 7
    SYNTHETIC_CATALOG = 8


class COLLECTION_TYPE(Enum):
    LIGHTCONE = 0
    STRUCTURE_COLLECTION = 1
    SIMULATION_COLLECTION = 2


class OpenTarget:
    def __init__(self, group: h5py.Group | h5py.File, header: OpenCosmoHeader):
        self.group = group
        self.header = header

    @property
    def data_type(self):
        return self.header.file.data_type


def get_file_type(file: h5py.File) -> FILE_TYPE:
    if "header" in file.keys():
        dtype = file["header"]["file"].attrs["data_type"]
        if dtype == "halo_particles":
            return FILE_TYPE.HALO_PARTICLES
        elif dtype == "halo_profiles":
            return FILE_TYPE.SOD_BINS
        elif dtype == "halo_properties":
            return FILE_TYPE.HALO_PROPERTIES
        elif dtype == "galaxy_properties":
            return FILE_TYPE.GALAXY_PROPERTIES
        elif dtype == "galaxy_particles":
            return FILE_TYPE.GALAXY_PARTICLES
        elif dtype == "diffsky_fits":
            return FILE_TYPE.SYNTHETIC_CATALOG
        else:
            raise ValueError(f"Unknown file type {dtype}")

    if not all("header" in group.keys() for group in file.values()):
        for subgroup in file.values():
            if not all("header" in g.keys() for g in subgroup.values()):
                raise ValueError(
                    "Unknown file type. "
                    "It appears to have multiple datasets, but organized incorrectly"
                )
    if all(group["header"]["file"].attrs["is_lightcone"] for group in file.values()):
        return FILE_TYPE.LIGHTCONE
    elif (
        len(set(group["header"]["file"].attrs["data_type"] for group in file.values()))
        == 1
    ):
        return FILE_TYPE.SIMULATION_COLLECTION

    elif all("data" not in group.keys() for group in file.values()):
        for group in file.values():
            sub_groups = {
                g["header"]["file"].attrs["data_type"]: g for g in group.values()
            }
            collection.structure.io.validate_linked_groups(sub_groups)
        return FILE_TYPE.SIMULATION_COLLECTION
    else:
        group = {name: group for name, group in file.items()}
        collection.structure.io.validate_linked_groups(group)
        return FILE_TYPE.STRUCTURE_COLLECTION


def make_all_targets(files: list[h5py.File]):
    targets: list[OpenTarget] = reduce(
        lambda targets, file: targets + make_file_targets(file), files, []
    )
    return targets


def make_file_targets(file: h5py.File):
    try:
        header = read_header(file, unit_convention=UnitConvention.COMOVING)
    except KeyError:
        header = None
    if header is not None and "data" in file.keys():
        return [OpenTarget(file, header)]
    if header is None and "data" in file.keys():
        raise ValueError(
            "This file appears to be missing a header. "
            "Are you sure it is an OpenCosmo file?"
        )
    if header is None:
        headers = {name: read_header(group) for name, group in file.items()}
    else:
        headers = {name: header for name in file.keys() if name != "header"}

    output = []
    for name, header in headers.items():
        target = OpenTarget(file[name], header)
        output.append(target)
    return output


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
    handles = [h5py.File(f) for f in file_list]
    file_types = list(map(get_file_type, handles))
    targets = make_all_targets(handles)
    targets = evaluate_load_conditions(targets, open_kwargs)
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
        idx_data = handle["index"]

        part = partition(comm, len(handler), idx_data, tree)
        if part is None:
            index = ChunkedIndex.empty()
        else:
            index = part.idx
            sim_region = part.region if part.region is not None else sim_region

    if metadata_group is not None:
        metadata_group = handle[metadata_group]

    state = dss.DatasetState.from_group(
        handle["data"],
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

    if header.file.is_lightcone and not bypass_lightcone:
        return collection.Lightcone({"data": dataset}, header.lightcone["z_range"])

    return dataset


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


def evaluate_load_conditions(targets: list[OpenTarget], open_kwargs: dict[str, bool]):
    """
    Datasets can define conditional loading via an addition group called "load/if".
    the "if" group can define parameters which must either be true or false for the
    given group to be loaded. These parameters can then be provided by the user to the
    "open" function. Parameters not specified by the user default to False.

    Note that some open kwargs may be used in other places in the opening process,
    and will just be ignored here.
    """
    if len(targets) == 1:
        return targets
    output = []
    for target in targets:
        try:
            ifgroup = target.group["load/if"]
        except KeyError:
            output.append(target)
            continue
        load = True
        for key, condition in ifgroup.attrs.items():
            load = load and (open_kwargs.get(key, False) == condition)
        if load:
            output.append(target)
    return output


def write(path: Path, dataset: Writeable, overwrite=False) -> None:
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

    schema = FileSchema()
    dataset_schema = dataset.make_schema()
    schema.add_child(dataset_schema, "root")

    if mpiio is not None:
        return mpiio.write_parallel(path, schema)

    file = h5py.File(path, "w")
    schema.allocate(file)

    writer = schema.into_writer()

    writer.write(file)
