from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

from opencosmo.index import empty, from_size, into_array, single_chunk
from opencosmo.mpi import get_comm_world, get_mpi, has_mpi

if TYPE_CHECKING:
    from opencosmo.collection.lightcone import Lightcone
    from opencosmo.dataset.dataset import Dataset
    from opencosmo.index import DataIndex, IndexArray


def get_random_take_index(
    n: int,
    ds_length: int,
    mode: Literal["local", "global"],
) -> DataIndex:
    if mode == "global" and has_mpi():
        return get_random_take_index_mpi(n, ds_length)

    if n > ds_length:
        return from_size(ds_length)

    generator = np.random.default_rng()
    rows = generator.choice(ds_length, n, replace=False)
    return apply_sort_index(rows)


def apply_sort_index(
    rows: DataIndex, sort_index: Optional[np.ndarray] = None
) -> np.ndarray:
    """Return row positions in ascending physical order.

    With sort_index: treats rows as logical sorted-order positions and maps
    them to physical row positions. Without sort_index: rows are already
    physical positions and are simply sorted.
    """
    arr = into_array(rows)
    if sort_index is not None:
        return np.sort(sort_index[arr])
    return np.sort(arr)


def _get_sort_index(ds: Dataset | Lightcone, sort_key: tuple[str, bool]) -> np.ndarray:
    sort_col, sort_desc = sort_key
    values = ds.select(sort_col).get_data("numpy", ignore_sort=True)
    assert isinstance(values, np.ndarray)
    if sort_desc:
        values = -values
    return np.argsort(values, kind="stable")


def get_rows_take_index(
    ds: Dataset | Lightcone, rows: DataIndex, sort_key: Optional[tuple[str, bool]]
) -> DataIndex:
    """Map user-provided logical (sorted-order) row positions to physical row positions."""
    if sort_key is None:
        return rows
    sort_index = _get_sort_index(ds, sort_key)
    return apply_sort_index(rows, sort_index)


def get_range_take_index(
    ds: Dataset | Lightcone,
    sort_key: Optional[tuple[str, bool]],
    start: int,
    size: int,
    mode: Literal["local", "global"],
):
    if mode == "global" and has_mpi():
        return get_range_take_index_mpi(ds, sort_key, start, size)

    ds_len = len(ds)
    if start + size > ds_len:
        size = ds_len - start

    if sort_key is not None:
        sort_index = _get_sort_index(ds, sort_key)
        return np.sort(sort_index[start : start + size])

    return single_chunk(start, size)


def get_end_take_index(
    n: int,
    ds: Dataset | Lightcone,
    sort_key,
    mode: Literal["local", "global"],
):
    ds_length = len(ds)
    if mode == "global" and has_mpi():
        comm = get_comm_world()
        assert comm is not None
        total_length = np.sum(comm.allgather(ds_length))
        if n > total_length:
            return from_size(ds_length)

        return get_range_take_index_mpi(ds, sort_key, total_length - n, n)

    start = ds_length - n
    if n > ds_length:
        start = 0
        n = ds_length

    if sort_key is not None:
        sort_index = _get_sort_index(ds, sort_key)
        return np.sort(sort_index[start : start + n])

    return single_chunk(start, n)


def get_range_take_index_mpi(
    ds: Dataset | Lightcone, sort_key: Optional[tuple[str, bool]], start: int, size: int
):
    comm = get_comm_world()
    assert comm is not None
    lengths = np.array(comm.allgather(len(ds)), dtype=np.int64)
    total_length = int(np.sum(lengths))

    if start > total_length:
        return empty()

    if start + size > total_length:
        size = total_length - start

    if sort_key is not None:
        global_sort_order = get_global_sort_order(ds, sort_key)

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


def get_global_sort_order(ds: Dataset | Lightcone, sort_key: tuple[str, bool]):
    comm = get_comm_world()
    assert comm is not None

    assert sort_key is not None
    sort_col, sort_desc = sort_key
    raw = ds.select(sort_col).get_data("numpy", ignore_sort=True)
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
        return from_size(ds_length)

    if comm.Get_rank() == 0:
        rng = np.random.default_rng()
        rows = np.sort(rng.choice(total_length, n, replace=False))
    else:
        rows = None
    return get_local_rows_simple(rows, lengths, comm)


def get_local_rows_simple(rows: IndexArray | None, lengths: list[int], comm):
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

        buffspec = [rows, scatter_lengths, buffer_offsets, get_mpi().INT64_T]
        comm.Scatterv(buffspec, local_rows)
    else:
        comm.Scatterv([None, None, None, get_mpi().INT64_T], local_rows)

    return local_rows - chunk_ranges[rank_num]
