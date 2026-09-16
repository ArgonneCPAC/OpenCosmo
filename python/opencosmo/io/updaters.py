from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from opencosmo.mpi import MPI


def offset_index(data: np.ndarray, offset: int = 0, comm: Optional[MPI.Comm] = None):
    valid = data >= 0
    result = np.full(len(data), -1, dtype=np.int64)
    if comm is not None:
        offsets = comm.allgather(offset)
        offset = np.sum(offsets[: comm.Get_rank()])

    result[valid] = data[valid] + offset
    return result


def do_idx_update(data: np.ndarray, comm: Optional[MPI.Comm] = None):
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
