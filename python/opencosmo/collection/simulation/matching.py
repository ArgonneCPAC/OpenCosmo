"""
Row matching for simulation collections. ``align`` realigns every target with
the active source through the match set's raw-row mappings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opencosmo.dataset import operations as dsops
from opencosmo.dataset import state as st
from opencosmo.dataset.state import DatasetState
from opencosmo.index import into_array
from opencosmo.index.mpi import is_in_global_index
from opencosmo.mapping.mapping import get_mapping
from opencosmo.mpi import get_comm_world, has_mpi

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opencosmo.collection.simulation.state import SimulationChild
    from opencosmo.mapping.mapping import DatasetMatchSet


def align(
    match_set: DatasetMatchSet, source: str, members: Mapping[str, SimulationChild]
) -> dict[str, SimulationChild]:
    """
    Realign every member with ``source``. Only rows of ``source`` that map into
    every target are kept, and each target is reordered to match them. Members
    keep their order.
    """
    datasets: dict[str, DatasetState] = {}
    for name, member in members.items():
        if not isinstance(member, DatasetState):
            raise RuntimeError(f"Matched simulation member {name} is not a dataset")
        datasets[name] = member

    aligned = __match_rows(match_set, datasets, source)
    return {name: aligned[name] for name in members}


def __match_rows(
    match_set: DatasetMatchSet, datasets: Mapping[str, DatasetState], source: str
) -> dict[str, DatasetState]:
    # Guard: active sort keys would cause dataset/state.py to re-sort the
    # carefully ordered rows produced below, silently misaligning everything.
    for name, ds in datasets.items():
        if ds.sort_key is not None and name != source:
            raise ValueError(
                f"Dataset '{name}' has an active sort key. Call match() before sort_by()."
            )

    reference_dataset = datasets[source]
    rows_to_keep = np.ones(len(reference_dataset), dtype=bool)
    index = reference_dataset.raw_index

    mappings = {}
    for name, dataset in datasets.items():
        if name == source:
            continue
        mapping = get_mapping(match_set, source, name, index)
        if mapping is None:
            raise ValueError(f"Unable to find mapping for dataset {dataset}")
        if not isinstance(mapping, np.ndarray):
            raise ValueError("Dataset matching does not support chunked mappings")
        rows_to_keep = rows_to_keep & (mapping >= 0)
        mappings[name] = mapping

    if has_mpi():
        return __match_rows_mpi(datasets, mappings, rows_to_keep, source)

    # The np.isin pass below is a required precondition for the unchecked
    # searchsorted in the final loop: it guarantees every value in `wanted`
    # is present in `target_index`, making the lookup safe without bounds checks.

    for name, mapping in mappings.items():
        mappings_to_keep = mapping[rows_to_keep]
        mappings_in_index = np.isin(
            mappings_to_keep, into_array(datasets[name].raw_index)
        )
        rows_to_keep[rows_to_keep] &= mappings_in_index

    # Must be taken only after the refinement above, so the source and every
    # target end up with the same number of rows.
    new_datasets = {
        source: dsops.take_rows(datasets[source], np.where(rows_to_keep)[0])
    }
    for name, mapping in mappings.items():
        target_index = into_array(datasets[name].raw_index)
        wanted = mapping[rows_to_keep]
        order = np.argsort(target_index, kind="stable")
        rows_to_take = order[np.searchsorted(target_index, wanted, sorter=order)]
        new_datasets[name] = dsops.take_rows(datasets[name], rows_to_take)

    return new_datasets


def __match_rows_mpi(
    datasets: Mapping[str, DatasetState],
    mappings: dict[str, np.ndarray],
    rows_to_keep: np.ndarray,
    source: str,
) -> dict[str, DatasetState]:
    """
    Match under MPI, where a rank pulls the GLOBAL target rows that its local
    source rows map to. Target rows therefore need not live on this rank, so the
    serial rank-local membership test does not apply.

    A row is still only matchable if it is present somewhere in the current
    global state. Rows dropped by an earlier match are gone from every rank and
    must not reappear, so membership is tested against the union of every rank's
    index rather than the local index alone.
    """

    comm = get_comm_world()
    assert comm is not None

    for name, mapping in mappings.items():
        rows_to_keep[rows_to_keep] &= is_in_global_index(
            mapping[rows_to_keep],
            into_array(datasets[name].raw_index),
            comm,
        )

    new_datasets = {
        source: dsops.take_rows(datasets[source], np.where(rows_to_keep)[0])
    }
    for name, mapping in mappings.items():
        dataset = datasets[name]
        new_datasets[name] = st.redistribute(dataset, mapping[rows_to_keep])
    return new_datasets
