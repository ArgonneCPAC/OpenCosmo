from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def n_in_range(
    index: NDArray[np.int_] | tuple,
    range_starts: int | NDArray[np.int_],
    range_sizes: int | NDArray[np.int_],
):
    range_starts = np.atleast_1d(range_starts)
    range_sizes = np.atleast_1d(range_sizes)
    match index:
        case np.ndarray():
            return __n_in_range_simple(index, range_starts, range_sizes)
        case (np.ndarray(), np.ndarray()):
            return __n_in_range_chunked(*index, range_starts, range_sizes)
        case _:
            raise ValueError(f"Unknown index type {type(index)}")


def __n_in_range_simple(
    index: NDArray[np.int_], start: NDArray[np.int_], size: NDArray[np.int_]
) -> NDArray[np.int_]:
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


@nb.njit
def __n_in_range_chunked(
    starts: NDArray[np.int_],
    sizes: NDArray[np.int_],
    range_starts: NDArray[np.int_],
    range_sizes: NDArray[np.int_],
) -> NDArray[np.int_]:
    """
    Return the number of elements in this index that fall within
    a specified data range. Used to mask spatial index.


    As with numpy, this is the half-open range [start, end)
    """
    if len(range_starts) != len(range_sizes):
        raise ValueError("Start and size arrays must have the same length")
    if np.any(range_sizes < 0):
        raise ValueError("Sizes must greater than or equal to zero")
    if len(starts) == 0:
        return np.zeros_like(range_starts)

    index_ranges = np.vstack((starts, starts + sizes))
    output = np.zeros_like(range_starts)
    for i in range(len(range_starts)):
        index_chunk_ranges = np.clip(
            index_ranges, a_min=range_starts[i], a_max=range_starts[i] + range_sizes[i]
        )
        output[i] = np.sum(index_chunk_ranges[1] - index_chunk_ranges[0])

    return output
