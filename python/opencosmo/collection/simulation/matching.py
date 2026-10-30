"""
Row matching for simulation collections. ``align`` realigns every target with
the active source through the match set's raw-row mappings.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, TypeGuard
from warnings import warn

import numpy as np

from opencosmo.collection.structure import operations as scops
from opencosmo.collection.structure.state import StructureState
from opencosmo.dataset import operations as dsops
from opencosmo.dataset.state import DatasetState
from opencosmo.index import into_array
from opencosmo.index.mpi import is_in_global_index
from opencosmo.mapping.mapping import get_mapping
from opencosmo.mpi import get_comm_world, has_mpi

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opencosmo.collection.simulation.state import SimulationChild
    from opencosmo.index import DataIndex
    from opencosmo.mapping.mapping import DatasetMatchSet


def is_structures(
    datasets: Mapping[str, DatasetState | StructureState],
) -> TypeGuard[dict[str, StructureState]]:
    return all(isinstance(state, StructureState) for state in datasets.values())


def is_datasets(
    datasets: Mapping[str, DatasetState | StructureState],
) -> TypeGuard[dict[str, DatasetState]]:
    return all(isinstance(state, DatasetState) for state in datasets.values())


def align(
    match_set: DatasetMatchSet,
    source: str,
    members: Mapping[str, SimulationChild],
) -> dict[str, SimulationChild]:
    """
    Realign every member with ``source``. Only rows of ``source`` that map into
    every target are kept, and each target is reordered to match them. Members
    keep their order.
    """
    datasets: dict[str, DatasetState | StructureState] = {}
    for name, member in members.items():
        if not isinstance(member, (DatasetState, StructureState)):
            raise TypeError("Matching is not available for lightcones")
        datasets[name] = member

    aligned = __match_rows(match_set, datasets, source)
    return {name: aligned[name] for name in members}


def __match_rows(
    match_set: DatasetMatchSet,
    datasets: Mapping[str, DatasetState | StructureState],
    source: str,
) -> dict[str, DatasetState | StructureState]:
    # Guard: active sort keys would cause dataset/state.py to re-sort the
    # carefully ordered rows produced below, silently misaligning everything.
    new_datasets: dict[str, DatasetState | StructureState] = {}
    for name, state in datasets.items():
        sort_key = (
            state.sort_key if isinstance(state, DatasetState) else state.source.sort_key
        )
        if sort_key is not None and name != source:
            sort_key = None
            warn(f"Dataset {name} is sorted. Matching will remove the sort")
        if isinstance(state, DatasetState):
            new_datasets[name] = replace(state, sort_key=sort_key)
        else:
            new_datasets[name] = replace(
                state,
                source=replace(state.source, sort_key=sort_key),
                linked_source=state.linked_source
                if state.linked_source is not None
                else state.source,
            )

    if not (is_datasets(new_datasets) or is_structures(new_datasets)):
        raise TypeError(
            "Matching requires members to be all datasets or all structure collections"
        )
    sources: dict[str, DatasetState] = {}
    for name, state in new_datasets.items():
        member_source = state if isinstance(state, DatasetState) else state.source
        if not isinstance(member_source, DatasetState):
            raise TypeError("Matching is not available for lightcone sources")
        sources[name] = member_source
    source_rows, target_rows = __match_dataset_rows(match_set, sources, source)

    # Under MPI target rows are global raw rows, serially they are local positions
    matched = {source: __with_rows(new_datasets[source], source_rows, False)}
    for name, rows in target_rows.items():
        matched[name] = __with_rows(new_datasets[name], rows, has_mpi())
    return matched


def __with_rows(
    member: DatasetState | StructureState, rows: DataIndex, redistribute: bool
) -> DatasetState | StructureState:
    if isinstance(member, DatasetState):
        if redistribute:
            return dsops.redistribute(member, rows)
        return dsops.take_rows(member, rows)
    if redistribute:
        return scops.redistribute(member, rows)
    return scops.take_rows(member, rows)


def __match_dataset_rows(
    match_set: DatasetMatchSet,
    datasets: Mapping[str, DatasetState],
    source: str,
) -> tuple[np.ndarray, dict[str, DataIndex]]:
    """
    Rows of ``source`` to keep, and for every target the rows aligned with them.
    """
    reference_dataset = datasets[source]
    rows_to_keep = np.ones(len(reference_dataset), dtype=bool)
    index = reference_dataset.raw_index

    mappings: dict[str, np.ndarray] = {}
    for name, dataset in datasets.items():
        if name == source:
            continue
        mapping = get_mapping(match_set, source, name, index)
        if mapping is None:
            raise ValueError(f"Unable to find mapping for dataset {dataset}")
        if not isinstance(mapping, np.ndarray):
            raise ValueError("Dataset matching does not support chunked mappings")  # noqa: TRY004 — capability limit, not a type error
        rows_to_keep = rows_to_keep & (mapping >= 0)
        mappings[name] = mapping

    if has_mpi():
        comm = get_comm_world()
        assert comm is not None
        # Target rows may live on any rank, but rows dropped by an earlier match
        # are gone everywhere and must not reappear.
        for name, mapping in mappings.items():
            rows_to_keep[rows_to_keep] &= is_in_global_index(
                mapping[rows_to_keep], into_array(datasets[name].raw_index), comm
            )
        return np.where(rows_to_keep)[0], {
            name: mapping[rows_to_keep] for name, mapping in mappings.items()
        }

    # The np.isin pass below is a required precondition for the unchecked
    # searchsorted in the final loop: it guarantees every value in `wanted`
    # is present in `target_index`, making the lookup safe without bounds checks.
    for name, mapping in mappings.items():
        mappings_to_keep = mapping[rows_to_keep]
        mappings_in_index = np.isin(
            mappings_to_keep, into_array(datasets[name].raw_index)
        )
        rows_to_keep[rows_to_keep] &= mappings_in_index

    target_rows: dict[str, DataIndex] = {}
    for name, mapping in mappings.items():
        target_index = into_array(datasets[name].raw_index)
        wanted = mapping[rows_to_keep]
        order = np.argsort(target_index, kind="stable")
        target_rows[name] = order[np.searchsorted(target_index, wanted, sorter=order)]
    return np.where(rows_to_keep)[0], target_rows
