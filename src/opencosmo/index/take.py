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


def take_chunked_from_chunked(from_: ChunkedIndex, by: ChunkedIndex):
    if from_.is_single_chunk() and from_.range()[0] == 0:
        return by

    from_idx = from_.into_array()
    output = np.zeros(len(by), dtype=int)
    output = __cfs_helper(from_idx, by.starts, by.sizes, output)
    diff = np.diff(output)
    breaks = np.where(diff != 1)[0]

    # Start indices of segments
    starts = output[np.r_[0, breaks + 1]]
    # End indices of segments (inclusive)
    ends = output[np.r_[breaks, output.size - 1]]

    sizes = ends - starts + 1
    return ChunkedIndex(starts, sizes)
