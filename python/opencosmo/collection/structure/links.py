"""
Link arithmetic for structure collections, on states.

A structure source is a sequence of leaves: one for a dataset, one per step
for a lightcone. Every link computation runs per leaf against that leaf's match
set and yields one index per leaf, which ``take_per_leaf`` applies to a target.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from opencosmo.collection.lightcone import operations as lcops
from opencosmo.collection.lightcone import state as lcst
from opencosmo.collection.structure import state as scst
from opencosmo.dataset import operations as dsops
from opencosmo.dataset.state import DatasetState
from opencosmo.index import coalesce_chunks, empty, into_array, offset
from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.io.writer import ColumnWriter
from opencosmo.mapping.mapping import (
    ChunkedSlot,
    get_mapping,
    get_slot_sizes,
    is_chunked_slot,
    rebuild_target_index,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from uuid import UUID

    from opencosmo.collection.lightcone.state import LeafKey, LightconeState
    from opencosmo.index import DataIndex
    from opencosmo.io.schema import Schema
    from opencosmo.mapping.mapping import DatasetMatchSet

    Source = DatasetState | LightconeState
    LinkTarget = DatasetState | LightconeState | scst.StructureState
    PerLeaf = dict[LeafKey, DataIndex]


LINK_ALIASES = {  # Left: Name in file, right: Name in collection
    "sodbighaloparticles_star_particles": "star_particles",
    "sodbighaloparticles_dm_particles": "dm_particles",
    "sodbighaloparticles_gravity_particles": "gravity_particles",
    "sodbighaloparticles_agn_particles": "agn_particles",
    "sodbighaloparticles_gas_particles": "gas_particles",
    "fofbighaloparticles_star_particles": "star_particles",
    "fofbighaloparticles_dm_particles": "dm_particles",
    "fofbighaloparticles_gravity_particles": "gravity_particles",
    "fofbighaloparticles_agn_particles": "agn_particles",
    "fofbighaloparticles_gas_particles": "gas_particles",
    "sod_profile": "halo_profiles",
    "haloprofiles": "halo_profiles",
    "galaxyproperties": "galaxy_properties",
    "galaxyparticles_star_particles": "star_particles",
}


def source_leaves(source: Source) -> tuple[tuple[LeafKey, DatasetState], ...]:
    if isinstance(source, DatasetState):
        return (((), source),)
    return source.leaves


def link_names(match_sets: Mapping[UUID, DatasetMatchSet]) -> tuple[str, ...]:
    """Collection-facing link names, sorted for determinism."""
    return tuple(
        sorted(
            {name for match_set in match_sets.values() for name in match_set.aliases}
        )
    )


def target_uuid(match_set: DatasetMatchSet, name: str) -> UUID:
    uuid = match_set.get_uuid(name)
    if uuid is None:
        raise ValueError(f"Unable to resolve link '{name}'")
    return uuid


def slot_values(
    match_sets: Mapping[UUID, DatasetMatchSet], source: Source, name: str
) -> tuple[np.ndarray, bool]:
    """
    Per-source-row slot values for link ``name`` and whether the link is chunked.

    Chunked links give the ``size`` column; simple links give the ``idx`` column
    with ``-1`` sentinels. Values follow the source's row iteration order: the
    sorted order of a dataset, and the physical leaf order of a lightcone.
    """
    values: list[np.ndarray] = []
    kinds: set[bool] = set()
    for _, leaf in source_leaves(source):
        leaf_values, chunked = __leaf_slot_values(match_sets[leaf.uuid], leaf, name)
        values.append(leaf_values)
        kinds.add(chunked)
    if len(kinds) != 1:
        raise ValueError(f"Link '{name}' has inconsistent slot kinds across steps")
    return np.concatenate(values), kinds.pop()


def resolve_index(
    match_sets: Mapping[UUID, DatasetMatchSet], source: Source, name: str
) -> PerLeaf:
    """Per-leaf target rows linked to the source's current rows."""
    per_leaf: PerLeaf = {}
    for key, leaf in source_leaves(source):
        match_set = match_sets[leaf.uuid]
        uuid = target_uuid(match_set, name)
        index = get_mapping(match_set, match_set.reference_source, uuid, leaf.raw_index)
        assert index is not None
        if not is_chunked_slot(match_set, uuid):
            index = np.asarray(index)
            index = index[index >= 0]
        per_leaf[key] = index
    return per_leaf


def rebuild_index(
    match_sets: Mapping[UUID, DatasetMatchSet],
    old_source: Source,
    new_source: Source,
    name: str,
) -> PerLeaf:
    """Per-leaf rows of targets aligned with ``old_source`` to keep for ``new_source``."""
    old_leaves = dict(source_leaves(old_source))
    per_leaf: PerLeaf = {}
    for key, new_leaf in source_leaves(new_source):
        old_leaf = old_leaves[key]
        match_set = match_sets[old_leaf.uuid]
        per_leaf[key] = rebuild_target_index(
            match_set,
            target_uuid(match_set, name),
            old_leaf.raw_index,
            new_leaf.raw_index,
        )
    return per_leaf


def resort_index(
    match_sets: Mapping[UUID, DatasetMatchSet], source: Source, name: str
) -> PerLeaf:
    """Per-leaf rows restoring targets to their source's file order."""
    per_leaf: PerLeaf = {}
    for key, leaf in source_leaves(source):
        values, chunked = __leaf_slot_values(match_sets[leaf.uuid], leaf, name)
        sort_index = __file_order(leaf)
        if not chunked:
            per_leaf[key] = sort_index[values >= 0]
            continue
        boundaries = np.zeros(len(values) + 1, dtype=np.int64)
        _ = np.cumsum(values, out=boundaries[1:])
        starts = boundaries[sort_index]
        sizes = values[sort_index]
        valid = sizes > 0
        per_leaf[key] = coalesce_chunks(starts[valid], sizes[valid])
    return per_leaf


def take_per_leaf(target: LinkTarget, per_leaf: PerLeaf) -> LinkTarget:
    """
    Apply per-leaf indices to a target aligned with the source's leaves. A
    lightcone target keeps exactly the steps present in ``per_leaf``.
    """
    if isinstance(target, DatasetState):
        return dsops.take_rows(target, per_leaf[()])
    if isinstance(target, scst.StructureState):
        rows = stacked_index(target.source, per_leaf)
        if isinstance(target.source, DatasetState):
            new_source: Source = dsops.take_rows(target.source, rows)
        else:
            new_source = lcops.take_rows(target.source, rows)
        linked_source = target.linked_source
        if linked_source is None:
            linked_source = target.source
        return replace(target, source=new_source, linked_source=linked_source)

    target_leaves = dict(target.leaves)
    missing = per_leaf.keys() - target_leaves.keys()
    if missing:
        raise RuntimeError(f"Linked lightcone is missing steps {sorted(missing)}")
    leaves = tuple(
        (key, dsops.take_rows(target_leaves[key], index))
        for key, index in per_leaf.items()
    )
    return lcst.from_leaves(
        leaves, target.maps, None, target.hidden, target.sort_key, target.scope
    )


def stacked_index(source: Source, per_leaf: PerLeaf) -> DataIndex:
    """Offset per-leaf indices into ``source``'s stacked row space."""
    if isinstance(source, DatasetState):
        return per_leaf.get((), empty())
    pieces: list[np.ndarray] = []
    running = 0
    for key, leaf in source.leaves:
        if (index := per_leaf.get(key)) is not None:
            pieces.append(into_array(offset(index, running)))
        running += len(leaf)
    if not pieces:
        return np.array([], dtype=np.int64)
    return np.concatenate(pieces)


def resolve(
    match_sets: Mapping[UUID, DatasetMatchSet],
    source: Source,
    linked: Mapping[str, LinkTarget],
) -> dict[str, LinkTarget]:
    """Align freshly opened link targets with their source."""
    names = set(link_names(match_sets))
    return {
        name: take_per_leaf(target, resolve_index(match_sets, source, name))
        if name in names
        else target
        for name, target in linked.items()
    }


def rebuild(
    match_sets: Mapping[UUID, DatasetMatchSet],
    old_source: Source,
    new_source: Source,
    linked: Mapping[str, LinkTarget],
) -> dict[str, LinkTarget]:
    """Re-align targets aligned with ``old_source`` to ``new_source``."""
    return {
        name: take_per_leaf(
            target, rebuild_index(match_sets, old_source, new_source, name)
        )
        for name, target in linked.items()
    }


def resort(
    match_sets: Mapping[UUID, DatasetMatchSet],
    source: Source,
    linked: Mapping[str, LinkTarget],
) -> dict[str, LinkTarget]:
    """Restore targets to file order when any source leaf is sorted."""
    if all(leaf.sort_key is None for _, leaf in source_leaves(source)):
        return dict(linked)
    return {
        name: take_per_leaf(target, resort_index(match_sets, source, name))
        for name, target in linked.items()
    }


def link_schema(
    match_sets: Mapping[UUID, DatasetMatchSet], source: Source, source_schema: Schema
) -> Schema:
    """Add source-row-aligned link datasets to a structure source schema."""
    if isinstance(source, DatasetState):
        return __with_data_linked(source_schema, match_sets[source.uuid], source)

    children = dict(source_schema.children)
    for key, leaf in source.leaves:
        step = key[0]
        child_names = [name for name in children if name.startswith(f"{step}_")]
        if len(child_names) != 1:
            raise RuntimeError(
                f"Unable to find source schema for lightcone step '{step}'"
            )
        child_name = child_names[0]
        children[child_name] = __with_data_linked(
            children[child_name], match_sets[leaf.uuid], leaf
        )
    return source_schema._replace(children=children)


def __with_data_linked(
    schema: Schema, match_set: DatasetMatchSet, leaf: DatasetState
) -> Schema:
    columns = {}
    for slot in match_set.primary_maps.values():
        datasets = (slot.start, slot.size) if isinstance(slot, ChunkedSlot) else (slot,)
        for dataset in datasets:
            name = dataset.name.rsplit("/", maxsplit=1)[-1]
            columns[name] = ColumnWriter.from_h5_dataset(
                dataset, leaf.raw_index, attrs=dict(dataset.attrs)
            )
    data_linked = make_schema(
        "data_linked",
        FileEntry.COLUMNS,
        columns={name: columns[name] for name in sorted(columns)},
    )
    return schema._replace(children=schema.children | {"data_linked": data_linked})


def __leaf_slot_values(
    match_set: DatasetMatchSet, leaf: DatasetState, name: str
) -> tuple[np.ndarray, bool]:
    uuid = target_uuid(match_set, name)
    if is_chunked_slot(match_set, uuid):
        return get_slot_sizes(match_set, uuid, leaf.raw_index), True
    mapping = get_mapping(match_set, match_set.reference_source, uuid, leaf.raw_index)
    assert mapping is not None
    return np.asarray(mapping), False


def __file_order(leaf: DatasetState) -> np.ndarray:
    """
    Positions in ``leaf``'s current (sorted) order of its rows in file order.

    Uses the leaf's raw row numbers rather than slot values: chunked slot sizes
    are not unique per row and cannot recover a permutation.
    """
    unsorted_rows = into_array(leaf.raw_data_handler.index)
    sorted_rows = into_array(leaf.raw_index)
    argsort_sorted_rows = np.argsort(sorted_rows)
    return argsort_sorted_rows[
        np.searchsorted(sorted_rows, unsorted_rows, sorter=argsort_sorted_rows)
    ]
