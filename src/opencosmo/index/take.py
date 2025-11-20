from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb  # type: ignore
import numpy as np
from numpy.typing import ArrayLike

from . import ChunkedIndex, SimpleIndex

if TYPE_CHECKING:
    from opencosmo.index.protocols import DataIndex


def take(from_: DataIndex, by: DataIndex):
    match (from_, by):
        case (SimpleIndex(), SimpleIndex()):
            return take_simple_from_simple(from_, by)
        case (SimpleIndex(), ChunkedIndex()):
            return take_chunked_from_simple(from_, by)
        case (ChunkedIndex(), SimpleIndex()):
            return take_simple_from_chunked(from_, by)
        case (ChunkedIndex(), ChunkedIndex()):
            return take_chunked_from_chunked(from_, by)


def take_simple_from_chunked(from_: ChunkedIndex, by: SimpleIndex):
    cumulative = np.insert(np.cumsum(from_.sizes), 0, 0)[:-1]
    arr = by.into_array()

    indices_into_chunks = np.argmax(arr[:, np.newaxis] < cumulative, axis=1) - 1
    output = arr - cumulative[indices_into_chunks] + from_.starts[indices_into_chunks]
    return SimpleIndex(output)


def take_simple_from_simple(from_: SimpleIndex, by: SimpleIndex):
    return SimpleIndex(from_.into_array()[by.into_array()])


def take_chunked_from_simple(from_: SimpleIndex, by: ChunkedIndex):
    from_arr = from_.into_array()
    starts = by.starts
    sizes = by.sizes
    output = np.zeros(sizes.sum(), dtype=int)
    output = __cfs_helper(from_arr, starts, sizes, output)
    return SimpleIndex(output)


@nb.njit
def __cfs_helper(arr, starts, sizes, storage):
    rs = 0
    for i in range(len(starts)):
        cstart = starts[i]
        csize = sizes[i]
        storage[rs : rs + csize] = arr[cstart : cstart + csize]
        rs += csize
    return storage


@nb.njit
def __cfc_helper(from_starts, from_sizes, by_starts, by_sizes):
    pass


@nb.njit
def prefix_sum(arr):
    out = np.empty(len(arr) + 1, dtype=arr.dtype)
    total = 0
    out[0] = 0
    for i in range(len(arr)):
        total += arr[i]
        out[i + 1] = total
    return out


@nb.njit
def find_chunk(prefix, x):
    """
    Returns index i such that prefix[i] <= x < prefix[i+1].
    """
    lo = 0
    hi = len(prefix) - 1  # prefix has length N+1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if prefix[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo


@nb.njit
def resolve_spanning_numba(
    start1, size1, start2, size2, out_start, out_size, out_owner
):
    """
    Resolves index2 slices into data-level chunks.
    Returns the number of output segments written.
    """
    prefix = prefix_sum(size1)
    out_pos = 0

    for j in range(len(start2)):
        logical = start2[j]
        remaining = size2[j]

        while remaining > 0:
            # Find which chunk in index1 we are inside
            i1 = find_chunk(prefix, logical)

            # Where inside that chunk?
            offset = logical - prefix[i1]

            # How many logical units remain in this chunk?
            chunk_left = size1[i1] - offset

            # How much we take
            take = chunk_left if chunk_left < remaining else remaining

            # Emit result
            out_start[out_pos] = start1[i1] + offset
            out_size[out_pos] = take
            out_owner[out_pos] = j

            out_pos += 1

            # Advance
            logical += take
            remaining -= take

    return out_pos


def take_chunked_from_chunked(from_: ChunkedIndex, by: ChunkedIndex):
    if from_.is_single_chunk() and from_.range()[0] == 0:
        return by

    max_out = len(by.sizes) * len(from_.sizes)
    out_start = np.empty(max_out, dtype=np.int64)
    out_size = np.empty(max_out, dtype=np.int64)
    out_owner = np.empty(max_out, dtype=np.int64)

    n = resolve_spanning_numba(
        from_.starts, from_.sizes, by.starts, by.sizes, out_start, out_size, out_owner
    )
    out_start = np.resize(out_start, (n,))
    out_size = np.resize(out_size, (n,))
    return ChunkedIndex(out_start, out_size)
