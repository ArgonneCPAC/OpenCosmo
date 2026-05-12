from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

import opencosmo.dataset.state as st
from opencosmo.index import single_chunk
from opencosmo.mpi import get_comm_world, get_mpi, has_mpi

if TYPE_CHECKING:
    from opencosmo.index import DataIndex, IndexArray


def get_random_take_index(
    n: int,
    ds_length: int,
    mode: Literal["local", "global"],
) -> DataIndex:
    if mode == "global" and has_mpi():
        return get_random_take_index_mpi(n, ds_length)

    if n > ds_length:
        raise ValueError("You cannot take more rows than exist in the dataset!")

    generator = np.random.default_rng()
    rows = generator.choice(ds_length, n, replace=False)
    return np.sort(rows)


def get_range_take_index(
    state: st.DatasetState,
    start: int,
    size: int,
    mode: Literal["local", "global"],
):
    if mode == "global" and has_mpi():
        return get_range_take_index_mpi(state, start, size)

    if start + size > len(state):
        raise ValueError("end must be less than the length of the dataset.")
    return single_chunk(start, size)


def get_range_take_index_mpi(state: st.DatasetState, start, size):
    comm = get_comm_world()
    assert comm is not None
    lengths = np.array(comm.allgather(len(state)), dtype=np.int64)
    total_length = int(np.sum(lengths))

    if start + size > total_length:
        raise ValueError(
            f"Tried to take {start + size} rows but total length of data is {total_length}"
        )

    if state.sort_key is not None:
        global_sort_order = get_global_sort_order(state)

        if comm.Get_rank() == 0:
            assert global_sort_order is not None
            n_ranks = comm.Get_size()
            chunk_ranges = np.zeros(n_ranks + 1, dtype=np.int64)
            chunk_ranges[1:] = np.cumsum(lengths)

            # Map each position in the global sort order to the rank that owns it.
            rank_of_element = np.searchsorted(
                chunk_ranges[1:], global_sort_order, side="right"
            )
            # Count how many sorted elements before `start` belong to each rank;
            # this is the local sorted-order start index for each rank's slice.
            lo_per_rank = np.bincount(rank_of_element[:start], minlength=n_ranks)
            # Count how many sorted elements in [start, start+size) belong to each rank.
            count_per_rank = np.bincount(
                rank_of_element[start : start + size], minlength=n_ranks
            )
        else:
            lo_per_rank = None
            count_per_rank = None

        lo_per_rank = comm.bcast(lo_per_rank)
        count_per_rank = comm.bcast(count_per_rank)

        rank = comm.Get_rank()
        local_start = int(lo_per_rank[rank])
        local_size = int(count_per_rank[rank])

        if local_size == 0:
            return np.array([], dtype=np.int64)
        return single_chunk(local_start, local_size)

    # Handle the case without sorting: contiguous global range
    rank = comm.Get_rank()
    offset = int(np.sum(lengths[:rank]))
    local_start = max(0, start - offset)
    local_end = min(int(lengths[rank]), start + size - offset)

    if local_end <= local_start:
        return np.array([], dtype=np.int64)
    return single_chunk(local_start, local_end - local_start)


def get_global_sort_order(state: st.DatasetState):
    comm = get_comm_world()
    assert comm is not None

    assert state.sort_key is not None
    sort_col, sort_desc = state.sort_key
    raw = st.get_data(st.select(state, {sort_col}), ignore_sort=True)[sort_col]
    local_values = np.asarray(
        raw.value if hasattr(raw, "value") else raw, dtype=np.float64
    )

    lengths = np.array(comm.allgather(len(local_values)), dtype=np.int64)
    total_length = int(np.sum(lengths))
    rank = comm.Get_rank()
    offset = int(np.sum(lengths[:rank]))

    # Use comm.Reduce to get the full catalog on rank 0.
    # Each rank writes its values at its global offset; summing gives the full array.
    local_contribution = np.zeros(total_length, dtype=np.float64)
    local_contribution[offset : offset + len(local_values)] = local_values
    recv = np.zeros(total_length, dtype=np.float64) if rank == 0 else None
    comm.Reduce(local_contribution, recv, op=get_mpi().SUM, root=0)

    # Other ranks return None
    if rank != 0:
        return None

    # Determine global sort range
    assert recv is not None
    global_sorted_phys = np.argsort(recv, kind="stable")
    if sort_desc:
        global_sorted_phys = global_sorted_phys[::-1]

    return global_sorted_phys


def get_random_take_index_mpi(n: int, ds_length: int):
    comm = get_comm_world()
    assert comm is not None
    lengths = comm.allgather(ds_length)

    if (total_length := np.sum(lengths)) < n:
        raise ValueError(
            f"Tried to take {n} rows but total length of data is {total_length}"
        )

    if comm.Get_rank() == 0:
        rng = np.random.default_rng()
        rows = np.sort(rng.choice(total_length, n, replace=False))
    else:
        rows = None
    return get_local_rows_simple(rows, lengths, comm)


def get_local_rows_simple(rows: IndexArray | None, lengths, comm):
    chunk_ranges = np.zeros(len(lengths) + 1, dtype=np.int64)
    chunk_ranges[1:] = np.cumsum(lengths)
    if comm.Get_rank() == 0:
        assert rows is not None
        chunk_ranges_in_index = np.searchsorted(rows, chunk_ranges)
        chunk_ranges_in_index = comm.bcast(chunk_ranges_in_index)
    else:
        chunk_ranges_in_index = comm.bcast(None)

    rank_num = comm.Get_rank()
    n_rows_local = chunk_ranges_in_index[rank_num + 1] - chunk_ranges_in_index[rank_num]

    local_rows = np.empty(n_rows_local, dtype=np.int64)

    if comm.Get_rank() == 0:
        scatter_lengths = chunk_ranges_in_index[1:] - chunk_ranges_in_index[:-1]
        buffer_offsets = np.zeros_like(scatter_lengths)
        buffer_offsets[1:] = np.cumsum(scatter_lengths)[:-1]

        buffspec = [rows, scatter_lengths, buffer_offsets, get_mpi().DOUBLE]
        comm.Scatterv(buffspec, local_rows)
    else:
        comm.Scatterv([None, None, None, get_mpi().DOUBLE], local_rows)

    return local_rows - chunk_ranges[rank_num]
