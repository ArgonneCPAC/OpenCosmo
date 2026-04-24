from __future__ import annotations

from collections import defaultdict
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import h5py
import healpy as hp
import numpy as np

import opencosmo as oc
from opencosmo import collection as occ
from opencosmo.dataset import state as st
from opencosmo.dataset.mpi import partition
from opencosmo.header import OpenCosmoHeader, read_header
from opencosmo.index.build import empty, from_range
from opencosmo.mpi import get_comm_world
from opencosmo.plugins.plugin import PluginType, apply_plugins
from opencosmo.spatial.builders import from_model
from opencosmo.spatial.region import FullSkyRegion, HealpixRegion
from opencosmo.spatial.tree import open_tree
from opencosmo.units import UnitConvention

if TYPE_CHECKING:
    from pathlib import Path

    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex

"""
This file contains all the internal logic for opening a file or files.

There are a few file structures we have to be able to support.

1. header + data groups -> single dataset
2. header + several non-data groups structure collection or lightcone collection. If lightcone collection,
   all dataset will have the same data type and will have is_lightcone set to true in the header.
3. no header, serveral groups -> Structure Collection


When a user passes multiple files, there are basically two options (at present), structure collection
or lightcone collection.

The former will consist of halo properties or galaxy properties, particles, and/or profiles
The later will consist of several datasets, each with the same data type and is_lightcone set to true
"""


class DatasetTarget(TypedDict):
    header: OpenCosmoHeader
    dataset_group: h5py.Group
    columns: list[h5py.Dataset]
    spatial_index: Optional[h5py.Group]


class FileType(Enum):
    DATASET = "dataset"
    LIGHTCONE = "lightcone"
    STRUCTURE_COLLECTION = "structure_collection"
    PARTICLES = "particles"
    SIMULATION_COLLECTION = "simulation_collection"


class CollectionType(Enum):
    pass


class FileTarget(TypedDict):
    dataset_group_types: dict[str, FileType]
    dataset_targets: list[DatasetTarget]
    dataset_groups: dict[str, list[DatasetTarget]]


def open_files(paths: list[Path], open_kwargs: dict[str, Any]):
    """
    Main back-end entry point for opening files.
    """

    func = partial(__make_file_target, open_kwargs=open_kwargs)
    targets = map(func, paths)

    valid_targets = [t for t in targets if t is not None]
    if not valid_targets:
        raise ValueError("No valid datasets found!")

    if len(valid_targets) > 1:
        collection_type = __determine_multi_file_collection_type(valid_targets)
        return collection_type.open(valid_targets, **open_kwargs)
    return __open_single_file(valid_targets[0])


def __make_group_map(group: h5py.File | h5py.Group, prefix: str = ""):
    index = {}
    for key, item in group.items():
        path = f"{prefix}/{key}"
        index[path] = item
        if isinstance(item, h5py.Group):
            index.update(__make_group_map(item, path))
    return index


def __make_file_target(path: Path, open_kwargs: dict[str, Any]) -> Optional[FileTarget]:
    """
    Search through the file for any valid datasets or dataset groups. For groups,
    identify the group types. Datasets with load conditions that are not
    met will be discarded.
    """
    file = h5py.File(path)
    group_map = __make_group_map(file)
    dataset_targets, group_targets = __find_all_datasets(group_map, open_kwargs)
    if not dataset_targets and not group_targets:
        return None
    group_types = __identify_group_types(dataset_targets, group_targets)
    return FileTarget(
        dataset_group_types=group_types,
        dataset_targets=dataset_targets,
        dataset_groups=group_targets,
    )


def __open_single_file(target: FileTarget) -> oc.Dataset | oc.collection.Collection:
    """
    Opens a single file, which may or may not contain
    several datasets
    """
    if len(target["dataset_targets"]) == 1:
        # Just one dataset, easy
        return open_single_dataset(target["dataset_targets"][0])

    elif target["dataset_targets"]:
        # Multiple datasets, but all grouped together
        if next(iter(target["dataset_group_types"].values())) == FileType.LIGHTCONE:
            # All lightcone datasets of the same type
            return occ.Lightcone.open([target])
        if (
            next(iter(target["dataset_group_types"].values()))
            == FileType.STRUCTURE_COLLECTION
        ):
            # Structure collection
            return occ.StructureCollection.open([target])
    elif target["dataset_groups"]:
        # Sometimes, lightcones have multiple datasets per slice
        if all(
            group_type == FileType.LIGHTCONE
            for group_type in target["dataset_group_types"].values()
        ):
            return occ.Lightcone.open([target])

        datasets = {
            name: __open_dataset_targets_for_sim_collection(
                targets, target["dataset_group_types"][name]
            )
            for name, targets in target["dataset_groups"].items()
        }
        if len(datasets) > 1:
            return occ.SimulationCollection(datasets)
        else:
            return next(iter(datasets.values()))
    raise ValueError(
        "Failed to open file. This is likely a bug. Please report it on github"
    )


def __open_dataset_targets_for_sim_collection(
    targets: list[DatasetTarget], group_type: FileType
):
    if len(targets) == 1:
        return open_single_dataset(targets[0])
    # Bad naming, will come back to this.
    file_target = FileTarget(
        dataset_group_types={"/": group_type},
        dataset_targets=targets,
        dataset_groups={},
    )
    match group_type:
        case FileType.STRUCTURE_COLLECTION:
            return occ.StructureCollection.open([file_target])
        # Currently the only nested collection we support, may
        # extend later
    raise ValueError(
        "File has an invalid structure. It looks like it should be a simulation collection, "
        "but the individual simulation datasets do not have the expected structure"
    )


def __determine_multi_file_collection_type(targets: list[FileTarget]):
    """
    When opening several files, the files must be composable into one of our
    supported collections. Here, we determine what the appropriate collection is.

    Most collections define their own opening logic, so we are free
    to simply delegate.
    """
    properties = []
    particles_or_profiles = []
    lightcones = []
    other_datasets = []
    # First, split files into their types

    for target in targets:
        if len(target["dataset_group_types"]) > 1:
            raise ValueError("Received an invalid combination of files!")

        file_type = next(iter(target["dataset_group_types"].values()))

        if file_type in [
            FileType.STRUCTURE_COLLECTION,
            FileType.SIMULATION_COLLECTION,
        ]:
            raise ValueError("Invalid combination of files!")
        if (
            file_type == FileType.DATASET
            and target["dataset_targets"][0]["header"].file.data_type == "halo_profiles"
        ):
            particles_or_profiles.append(target)
        elif file_type == FileType.DATASET and target["dataset_targets"][0][
            "header"
        ].file.data_type in ["halo_properties", "galaxy_properties"]:
            properties.append(target)
        elif file_type == FileType.PARTICLES:
            particles_or_profiles.append(target)
        elif file_type == FileType.LIGHTCONE:
            lightcones.append(target)
        elif file_type == FileType.DATASET:
            other_datasets.append(target)
        else:
            raise ValueError("Invalid combination of files!")

    return __get_collection_type_from_categorized_lists(
        properties, particles_or_profiles, lightcones, other_datasets
    )


def __get_collection_type_from_categorized_lists(
    properties: list[FileTarget],
    particles_or_profiles: list[FileTarget],
    lightcones: list[FileTarget],
    other_datasets: list[FileTarget],
):
    """
    Determines the collection type from a categorized
    list of files
    """
    flags = (
        len(properties) > 0,
        len(particles_or_profiles) > 0,
        len(lightcones) > 0,
        len(other_datasets) > 0,
    )
    match flags:
        case (True, True, False, False):
            return occ.StructureCollection
        case (False, False, True, False):
            return occ.Lightcone
        case (True, False, False, False):
            return __get_multi_dataset_type(properties)
        case (False, False, False, True):
            return __get_multi_dataset_type(other_datasets)
        case (False, True, True, False):
            # A single property dataset on a lightcone will be categorized as a lightcone
            # The StructureCollection will through an error if there is a weirder setup
            return occ.StructureCollection
        case _:
            raise ValueError("Invalid combination of files")


def __get_multi_dataset_type(file_targets: list[FileTarget]):
    """
    If you have multiple datasets of the same type, we have to figure
    out how to open them
    """
    dtypes = set(
        ft["dataset_targets"][0]["header"].file.data_type for ft in file_targets
    )
    is_lightcone = set(
        ft["dataset_targets"][0]["header"].file.is_lightcone for ft in file_targets
    )
    if dtypes == {"halo_properties", "galaxy_properties"}:  # special case
        return occ.StructureCollection

    if len(dtypes) == 1 or len(is_lightcone) > 1:
        raise ValueError(
            "When opening multiple files, they must either be several different data types from a single simulation, "
            "a single data type from several simulations, or a single lightcone data type from a single simulation"
        )
    if is_lightcone.pop():
        return occ.Lightcone
    else:
        return occ.SimulationCollection


def __identify_group_types(
    ds_targets: list[DatasetTarget], group_targets: dict[str, list[DatasetTarget]]
):
    """
    Figure out what our datasets should combine into
    """
    if group_targets:
        return {
            name: __identify_group_types(targets, {})["/"]
            for name, targets in group_targets.items()
        }

    data_types = set(str(t["header"].file.data_type) for t in ds_targets)
    is_lightcone = [t["header"].file.is_lightcone for t in ds_targets]

    if all("particle" in dt for dt in data_types):  # particles
        return {"/": FileType.PARTICLES}
    if len(data_types) == 1 and all(is_lightcone):  # lightcone
        return {"/": FileType.LIGHTCONE}
    if len(ds_targets) == 1:  # Just a dataset
        return {"/": FileType.DATASET}

    parents = set(t["dataset_group"].parent.name for t in ds_targets)
    if (
        len(parents) == 1 and len(data_types) > 1
    ):  # Multiple data types, but not all particles
        return {"/": FileType.STRUCTURE_COLLECTION}
    return {"/": FileType.SIMULATION_COLLECTION}  # Organized into multiple groups


def __find_all_headers(file_map: dict):
    return list(filter(lambda key: key.endswith("header"), file_map.keys()))


def __find_all_datasets(
    file_map: dict[str, h5py.File | h5py.Group | h5py.Dataset], open_kwargs
) -> tuple[list[DatasetTarget], dict[str, list[DatasetTarget]]]:
    """
    Search through a file and locate all the datasets. Each dataset is identified
    with a "data" group. The header associated with the file is the closest
    header at the same level or above.

    However datasets that are not at the top level need to be grouped. Currently, we
    have the following options.

    1. The datasets are are lightcone datasets, all have the same type, and all come from the
       same simulation. These should be grouped and opened as a lightcone.
    2. Otherwise, we're talking about a simulation collection.
    """

    known_headers = __find_all_headers(file_map)

    if not known_headers:
        raise ValueError(
            f"The file at {next(iter(file_map.values())).file.filename}, does not appear to be an OpenCosmoFile"
        )

    all_file_headers: list[OpenCosmoHeader] = list(
        map(
            lambda header_group: read_header(file_map[header_group].parent),
            known_headers,
        )
    )
    if len(all_file_headers) > 1:
        known_datasets, known_dataset_groups = __get_collection_dataset_groups(
            file_map, known_headers, all_file_headers, open_kwargs
        )

    else:
        known_datasets = __find_datasets_under_group(
            file_map[known_headers[0]].parent.name,
            file_map,
            all_file_headers[0],
            open_kwargs,
        )
        known_dataset_groups = {}

    if not known_datasets and not known_dataset_groups:
        raise ValueError(
            f"File {next(iter(file_map.values())).file.filename} contains an OpenCosmo header, but does not seem to be formatted correctly!"
        )
    return known_datasets, known_dataset_groups


def __find_datasets_under_group(
    group_name: str, file_map, header: OpenCosmoHeader, open_kwargs: dict[str, Any]
):
    """
    Given a header and the group it lives in, find all datasets
    that live at the same level or below that header.
    """
    known_datasets = []
    if group_name != "/":
        group_name = f"{group_name}/"

    known_dataset_groups = list(
        filter(
            lambda key: key.startswith(f"{group_name}") and key.endswith("/data"),
            file_map.keys(),
        )
    )

    for ds_group_name in known_dataset_groups:
        ds_group_parent = ds_group_name.rsplit("/", maxsplit=1)[0]
        ds_group_parent += "/"

        columns = [
            nds_[1]
            for nds_ in filter(
                lambda nds: (
                    ds_group_parent in nds[0]
                    and isinstance(nds[1], h5py.Dataset)
                    and "header" not in nds[0]
                    and f"{ds_group_parent}index" not in nds[0]
                ),
                file_map.items(),
            )
        ]
        index_group = file_map.get(f"{ds_group_parent}index")

        target = DatasetTarget(
            header=header,
            dataset_group=file_map[ds_group_name].parent,
            columns=columns,
            spatial_index=index_group,
        )
        if evaluate_load_conditions(target, open_kwargs):
            known_datasets.append(target)

    return known_datasets


def __get_collection_dataset_groups(file_map, header_groups, headers, open_kwargs):
    """
    If a file has multiple headers, we may need to keep the datasets under each header
    seperate until we combine them later. It's also possible we are working with a
    structure collection
    """
    dataset_groups = {}
    all_datasets = []
    for group, header in zip(header_groups, headers):
        dataset_targets = __find_datasets_under_group(
            file_map[group].parent.name, file_map, header, open_kwargs
        )
        all_datasets += dataset_targets
        if dataset_targets:
            dataset_groups[group] = dataset_targets

    if dataset_groups:
        dataset_groups = __combine_dataset_groups(dataset_groups)
    data_types = set(t["header"].file.data_type for t in all_datasets)
    parent_groups = set([t["dataset_group"].parent for t in all_datasets])

    if (
        len(data_types) > 1 and len(parent_groups) == 1
    ):  # Only possible for a structure collection
        return all_datasets, {}

    return [], dataset_groups


def __combine_dataset_groups(groups: dict[str, list[DatasetTarget]]):
    """
    The only context in which datasets are nested two layers deep is if we have a simulation
    collection with a structure collection inside. This function checks for this case and
    combines those nested datasets into groups.
    """
    output_groups = defaultdict(list)
    for group_name, datasets in groups.items():
        output_group_name = group_name.split("/")[1]
        output_groups[output_group_name].extend(datasets)

    return output_groups


def open_single_dataset(
    target: DatasetTarget,
    metadata_group: Optional[str] = None,
    bypass_lightcone: bool = False,
    bypass_mpi: bool = False,
):
    header = target["header"]
    ds_group = target["dataset_group"]
    columns = target["columns"]

    assert header is not None

    try:
        box_size = header.with_units("scalefree").simulation["box_size"].value
    except AttributeError:
        box_size = None

    if target["spatial_index"] is not None:
        tree = open_tree(
            target["spatial_index"],
            box_size,
            header.file.is_lightcone,
        )
    else:
        tree = None

    if header.file.region is not None:
        sim_region = from_model(header.file.region)
    elif header.file.is_lightcone and tree is not None:
        pixels = tree.get_full_index(tree.max_level)
        sim_region = HealpixRegion(pixels, nside=2**tree.max_level)
    elif header.file.data_type == "healpix_map":
        assert header.healpix_map["full_sky"]
        sim_region = FullSkyRegion()
    elif not header.file.is_lightcone:
        p1 = (0, 0, 0)
        p2 = tuple(header.simulation["box_size"].value for _ in range(3))
        sim_region = oc.make_box(p1, p2)

    index: Optional[DataIndex] = None
    ds_length = len(next(iter(columns)))

    if not bypass_mpi and (comm := get_comm_world()) is not None:
        assert partition is not None
        try:
            idx_data = ds_group["index"]
            part = partition(comm, ds_length, idx_data, tree)
            if part is None:
                index = empty()
            else:
                index = part.idx
                sim_region = part.region if part.region is not None else sim_region
            if header.file.is_lightcone:
                sim_region = __expand_lightcone_region(sim_region, tree)

        except KeyError:
            n_ranks = comm.Get_size()
            n_per = ds_length // n_ranks
            chunk_boundaries = [i * n_per for i in range(n_ranks + 1)]
            chunk_boundaries[-1] = ds_length
            rank = comm.Get_rank()
            index = from_range(chunk_boundaries[rank], chunk_boundaries[rank + 1])

    state = st.DatasetState.from_target(
        target,
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
        return __open_healpix_map(dataset, sim_region)
    elif header.file.is_lightcone and not bypass_lightcone:
        return occ.Lightcone.from_datasets(
            {"data": dataset}, header.lightcone["z_range"]
        )

    return apply_plugins(PluginType.DatasetOpen, dataset)


def __open_healpix_map(dataset: oc.Dataset, sim_region):
    header = dataset.header
    if (comm := get_comm_world()) is not None and isinstance(
        sim_region, HealpixRegion
    ):  # partitioning has to be done manually since we don't store a spatial index
        pixels = sim_region.pixels
        splits = np.array(comm.allgather(len(dataset)))
        splits = np.insert(np.cumsum(splits), 0, 0)
        rank = comm.Get_rank()
        sim_region = HealpixRegion(
            pixels[splits[rank] : splits[rank + 1]],
            nside=header.healpix_map["nside"],
        )
    elif isinstance(sim_region, FullSkyRegion) or header.healpix_map["full_sky"]:
        sim_region = HealpixRegion(dataset.index, nside=header.healpix_map["nside"])

    return occ.HealpixMap(
        {"data": dataset},
        header.healpix_map["nside"],
        header.healpix_map["nside_lr"],
        header.healpix_map["ordering"],
        header.healpix_map["full_sky"],
        header.healpix_map["z_range"],
        region=sim_region,
    )


def __expand_lightcone_region(region, tree):
    pixels = region.pixels
    npix_ratio = hp.nside2npix(2**tree.max_level) // hp.nside2npix(region.nside)
    pixels = pixels[:, None] * npix_ratio + np.arange(npix_ratio)
    pixels = pixels.flatten()

    full_pixels = tree.get_full_index(tree.max_level)
    full_pixels = np.intersect1d(pixels, full_pixels)
    return HealpixRegion(full_pixels, 2**tree.max_level)


def evaluate_load_conditions(
    target: DatasetTarget, open_kwargs: dict[str, bool]
) -> bool:
    """
    Datasets can define conditional loading via an addition group called "load/if".
    the "if" group can define parameters which must either be true or false for the
    given group to be loaded. These parameters can then be provided by the user to the
    "open" function. Parameters not specified by the user default to False.

    Note that some open kwargs may be used in other places in the opening process,
    and will just be ignored here.
    """
    try:
        ifgroup = target["dataset_group"]["load/if"]
    except KeyError:
        return True
    load = True
    for key, condition in ifgroup.attrs.items():
        load = load and (open_kwargs.get(key, False) == condition)
    return load
