from __future__ import annotations

import numba as nb  # type: ignore
import numpy as np

SimpleIndex = np.ndarray
ChunkedIndex = tuple[np.ndarray, np.ndarray]


def take(from_, by):
    match (from_, by):
        case (np.ndarray(), np.ndarray()):
            return __take_simple_from_simple(from_, by)
        case (np.ndarray(), (np.ndarray(), np.ndarray())):
            return __take_chunked_from_simple(from_, by)
        case ((np.ndarray(), np.ndarray()), np.ndarray()):
            return __take_simple_from_chunked(from_, by)
        case ((np.ndarray(), np.ndarray()), (np.ndarray(), np.ndarray())):
            return __take_chunked_from_chunked(from_, by)


def __take_simple_from_chunked(from_: ChunkedIndex, by: SimpleIndex):
    cumulative = np.insert(np.cumsum(from_[1]), 0, 0)[:-1]

    indices_into_chunks = np.argmax(by[:, np.newaxis] < cumulative, axis=1) - 1
    output = by - cumulative[indices_into_chunks] + from_[0][indices_into_chunks]
    return output


def __take_simple_from_simple(from_: np.ndarray, by: np.ndarray):
    return from_[by]


def __take_chunked_from_simple(from_: SimpleIndex, by: ChunkedIndex):
    output = np.zeros(by[1].sum(), dtype=int)
    output = __cfs_helper(from_, *by, output)
    return output


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


def __take_chunked_from_chunked(from_: ChunkedIndex, by: ChunkedIndex):
    if len(from_[0]) == 0 and from_[0][0] == 0:
        return by

    max_out = len(by[1]) * len(from_[1])
    out_start = np.empty(max_out, dtype=np.int64)
    out_size = np.empty(max_out, dtype=np.int64)
    out_owner = np.empty(max_out, dtype=np.int64)

    n = resolve_spanning_numba(
        from_[0], from_[1], by[0], by[1], out_start, out_size, out_owner
    )
    out_start = np.resize(out_start, (n,))
    out_size = np.resize(out_size, (n,))
    return out_start, out_size
