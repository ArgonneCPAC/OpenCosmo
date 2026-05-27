from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opencosmo._lib import index as idx

if TYPE_CHECKING:
    from opencosmo.index import DataIndex, IndexArray, SimpleIndex


def n_in_range(
    index: DataIndex,
    range_starts: int | IndexArray,
    range_sizes: int | IndexArray,
) -> IndexArray:
    range_starts = np.atleast_1d(range_starts)
    range_sizes = np.atleast_1d(range_sizes)
    match index:
        case np.ndarray():
            return __n_in_range_simple(index, range_starts, range_sizes)
        case (np.ndarray(), np.ndarray()):
            return idx.n_in_range_chunked(index[0], index[1], range_starts, range_sizes)
        case _:
            raise ValueError(f"Unknown index type {type(index)}")


def __n_in_range_simple(
    index: SimpleIndex, start: IndexArray, size: IndexArray
) -> IndexArray:
    if len(start) != len(size):
        raise ValueError("Start and size arrays must have the same length")
    if np.any(size < 0):
        raise ValueError("Sizes must greater than or equal to zero")
    if len(index) == 0:
        return np.zeros_like(start)

    ends = start + size
    index_to_search = np.sort(index)
    start_idxs = np.searchsorted(index_to_search, start, "left")
    end_idxs = np.searchsorted(index_to_search, ends, "left")
    return end_idxs - start_idxs
