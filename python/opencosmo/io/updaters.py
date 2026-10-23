from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from opencosmo.mpi import MPI


def do_idx_update(data: np.ndarray, comm: MPI.Comm | None = None):
    # An idx metadata column links each structure to at most one row in a target
    # dataset, using -1 to mark structures with no linked row (e.g. halos without
    # a profile). The target dataset is written containing only the linked rows,
    # in structure order, so the rewritten idx must give each linked structure a
    # contiguous 0-based index while preserving the -1 sentinels. Under MPI the
    # target is concatenated across ranks, so each rank offsets its indices by the
    # number of linked rows on the ranks before it.
    valid = data >= 0
    n_valid = int(valid.sum())
    if comm is None:
        offset = 0
    else:
        counts = comm.allgather(n_valid)
        offset = int(np.sum(counts[: comm.Get_rank()]))
    result = np.full(len(data), -1, dtype=np.int64)
    result[valid] = np.arange(offset, offset + n_valid)
    return result


def do_start_update(data: np.ndarray, size: np.ndarray, comm: MPI.Comm | None):
    psum = np.insert(np.cumsum(size), 0, 0)[:-1]
    if comm is None:
        return psum
    lengths = comm.allgather(np.sum(size))
    offsets = np.insert(np.cumsum(lengths), 0, 0)
    offset = offsets[comm.Get_rank()]
    return psum + offset
