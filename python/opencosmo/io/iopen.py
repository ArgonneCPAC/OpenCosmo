from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import healpy as hp
import numpy as np

import opencosmo as oc
from opencosmo import collection as occ
from opencosmo.dataset import state as st
from opencosmo.header import OpenCosmoHeader
from opencosmo.io import plan
from opencosmo.io.discover import discover_all, is_particle_group
from opencosmo.io.specs import group_by_scope, match_spec
from opencosmo.mpi import get_comm_world
from opencosmo.plugins.contexts import DatasetOpenCtx, HookPoint
from opencosmo.plugins.hook import fold
from opencosmo.spatial.builders import from_model
from opencosmo.spatial.region import FullSkyRegion, HealpixRegion
from opencosmo.spatial.tree import open_tree
from opencosmo.units import UnitConvention

if TYPE_CHECKING:
    from pathlib import Path

    import h5py

    from opencosmo.header import OpenCosmoHeader
    from opencosmo.io.index_spec import IndexSpec
    from opencosmo.io.io import MpiMode

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


def open_files(
    paths: list[Path],
    open_kwargs: dict[str, Any],
    mpi_mode: "MpiMode | None" = None,
) -> oc.Dataset | oc.collection.Collection:
    """
    Main back-end entry point for opening files.

    Pipeline: discover metadata for every file (one collective allgather), group the
    discovered layouts by governing header scope, then for each scope match a spec,
    verify it, distribute files across ranks, and build this rank's object. A single
    scope returns its object directly; multiple scopes wrap in a SimulationCollection.
    Every rank computes identical scopes/specs/assignments from identical, path-sorted
    layouts, so the only collective in the whole path is the discovery allgather.
    """
    if mpi_mode is None:
        from opencosmo.io.io import MpiMode

        mpi_mode = MpiMode.SPATIAL

    comm = get_comm_world()
    rank = comm.Get_rank() if comm is not None else 0
    nranks = comm.Get_size() if comm is not None else 1

    layouts = discover_all(paths, comm)

    # Discovery captures structural failures into FileLayout.error rather than raising
    # mid-collective. Every rank holds the identical errored set (allgather), so raising
    # here is collective. The message includes each bad path (test_open_bad_data asserts
    # the offending path appears in the ValueError).
    errored = [fl for fl in layouts if fl.error is not None]
    if errored:
        details = "\n".join(f"  {fl.path}: {fl.error}" for fl in errored)
        raise ValueError(f"Failed to open one or more files:\n{details}")

    # Particle files carry a "*_particles" data_type in their header. They only make
    # sense when opened alongside the properties dataset that links to them (a
    # StructureCollection). Opening particles on their own is unsupported: if every
    # discovered group is a particle group, there is nothing to anchor them, so fail
    # loudly rather than fabricate a SimulationCollection of orphaned particle types.
    groups = [g for fl in layouts if fl.error is None for g in fl.groups]
    if groups and all(is_particle_group(g) for g in groups):
        raise ValueError(
            "Cannot open particle data on its own. Particle datasets must be opened "
            "together with their properties dataset (e.g. halo or galaxy properties), "
            "which links to them as a StructureCollection."
        )

    scopes = group_by_scope(layouts)
    if not scopes:
        raise ValueError("No valid datasets found!")

    children: dict[str, oc.Dataset | oc.collection.Collection] = {}
    for scope_name in sorted(scopes):
        sub = scopes[scope_name]
        spec = match_spec(sub)
        if spec is None:
            raise ValueError(
                "Failed to open file. This is likely a bug. Please report it on github"
            )
        spec.verify(sub)
        assignments = plan.distribute(sub, mpi_mode, nranks)
        child = plan.build_from_assignment(assignments[rank], sub, spec, open_kwargs)
        # A scope whose every dataset was gated out by load/if conditions builds to
        # None (see plan.build_from_assignment); drop it.
        if child is not None:
            children[scope_name] = child

    if not children:
        raise ValueError("No valid datasets found!")
    if len(children) == 1:
        return next(iter(children.values()))
    return occ.SimulationCollection(children)


def open_dataset(
    target: DatasetTarget,
    index: "IndexSpec",
    *,
    metadata_group: Optional[str] = None,
    open_kwargs: dict[str, Any] = {},
) -> oc.Dataset:
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

    ds_length = len(next(iter(columns)))
    comm = get_comm_world()
    data_index, sim_region = index(comm, header, ds_group, tree, ds_length, sim_region)

    state = st.state_from_target(
        target,
        UnitConvention.COMOVING,
        sim_region,
        open_kwargs,
        data_index,
        metadata_group,
    )

    dataset = oc.Dataset(
        header,
        state,
        tree=tree,
    )
    dataset = fold(HookPoint.DatasetOpen, DatasetOpenCtx(dataset, open_kwargs)).dataset
    return dataset


def _open_healpix_map(dataset: oc.Dataset, sim_region):
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


def _expand_lightcone_region(region, tree):
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
