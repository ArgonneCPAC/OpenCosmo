from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

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
