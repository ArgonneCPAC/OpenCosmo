from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import h5py
import healpy as hp
import numpy as np

import opencosmo as oc
from opencosmo import collection as occ
from opencosmo.collection.structure import structure as sc
from opencosmo.dataset import state as st
from opencosmo.dataset.mpi import partition
from opencosmo.header import OpenCosmoHeader, read_header
from opencosmo.index.build import empty, from_range
from opencosmo.io.openers import TargetSummary
from opencosmo.mpi import get_comm_world
from opencosmo.plugins.contexts import DatasetOpenCtx, HookPoint
from opencosmo.plugins.hook import fold
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


class FileTarget(TypedDict):
    dataset_targets: list[DatasetTarget]
    dataset_groups: dict[str, list[DatasetTarget]]


class _SingleDatasetOpener:
    """Opener for single-dataset files."""

    @classmethod
    def claim(cls, summary: TargetSummary) -> bool:
        """Claim if there's exactly one file with exactly one dataset."""
        return (
            summary.total_targets == 1
            and len(summary.dataset_targets) == 1
            and len(summary.dataset_groups) == 0
        )

    @classmethod
    def open(
        cls, targets: list[FileTarget], **kwargs: bool
    ) -> oc.Dataset | oc.collection.Collection:
        """Open a single dataset."""
        target = targets[0]
        return open_single_dataset(target["dataset_targets"][0], open_kwargs=kwargs)


def __get_openers() -> list[type]:
    """Return the ordered list of openers to try."""
    # Import at runtime to avoid circular imports
    from opencosmo.collection.lightcone import lightcone as lc
    from opencosmo.collection.simulation import simulation as sim

    return [
        _SingleDatasetOpener,
        sc.StructureCollection,
        lc.Lightcone,
        sim.SimulationCollection,
    ]


def open_files(paths: list[Path], open_kwargs: dict[str, Any]):
    """
    Main back-end entry point for opening files.
    """

    func = partial(__make_file_target, open_kwargs=open_kwargs)
    targets = map(func, paths)

    valid_targets = [t for t in targets if t is not None]
    if not valid_targets:
        raise ValueError("No valid datasets found!")

    summary = TargetSummary.build(valid_targets)

    for opener_cls in __get_openers():
        if opener_cls.claim(summary):  # type: ignore[attr-defined]
            return opener_cls.open(valid_targets, **open_kwargs)  # type: ignore[attr-defined]

    raise ValueError("Invalid combination of files")


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
    Search through the file for any valid datasets or dataset groups.
    Datasets with load conditions that are not met will be discarded.
    """
    file = h5py.File(path)
    group_map = __make_group_map(file)
    dataset_targets, group_targets = __find_all_datasets(group_map, open_kwargs)
    if not dataset_targets and not group_targets:
        return None
    return FileTarget(
        dataset_targets=dataset_targets,
        dataset_groups=group_targets,
    )


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
    open_kwargs: dict[str, Any] = {},
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
        pixels = tree.get_partitions_with_data(tree.max_level)
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
            part = partition(comm, header, ds_group["index"], ds_group["data"], tree)
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

    state = st.state_from_target(
        target,
        UnitConvention.COMOVING,
        sim_region,
        open_kwargs,
        index,
        metadata_group,
    )

    dataset = oc.Dataset(
        header,
        state,
        tree=tree,
    )
    dataset = fold(HookPoint.DatasetOpen, DatasetOpenCtx(dataset, open_kwargs)).dataset
    if header.file.data_type == "healpix_map":
        return __open_healpix_map(dataset, sim_region)
    elif header.file.is_lightcone and not bypass_lightcone:
        return occ.Lightcone.from_datasets(
            {0: dataset}, header.lightcone["z_range"], **open_kwargs
        )

    return dataset


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

    full_pixels = tree.get_partitions_with_data(tree.max_level)
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
