from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opencosmo.index._ops import chunked_into_array

if TYPE_CHECKING:
    from numpy.typing import NDArray

from opencosmo.index.unary import get_length


def mask(index, boolean_mask):
    match index:
        case np.ndarray():
            return __mask_simple(index, boolean_mask)
        case (np.ndarray(), np.ndarray()):
            return __mask_chunked(index, boolean_mask)
        case _:
            raise TypeError(f"Unknown index type {type(index)}")


def __mask_simple(index: NDArray[np.int_], boolean_mask: NDArray[np.bool_]):
    if (lm := len(boolean_mask)) > len(index):
        raise ValueError(
            "Boolean mask must be less than or equal to the length of the index itself"
        )

    return index[:lm][boolean_mask]


def __mask_chunked(index: tuple, boolean_mask: NDArray[np.bool_]):
    array = into_array(index)
    return array[boolean_mask]


def into_array(index: np.ndarray | tuple):
    if get_length(index) == 0:
        return np.array([], dtype=np.int64)

    match index:
        case np.ndarray():
            return index
        case (np.ndarray(), np.ndarray()):
            if len(index[0]) == 1:
                return np.arange(index[0][0], index[0][0] + index[1][0])

            return __chunked_into_array(*index)


def __chunked_into_array(starts: NDArray[np.int_], sizes: NDArray[np.int_]):
    print("Using C!")
    if starts.ndim != 1 or sizes.ndim != 1:
        raise ValueError("Indicies must be 1 dimension")
    if len(starts) != len(sizes):
        raise ValueError("Indicies must be 1 dimension")
    return chunked_into_array(starts, sizes)
