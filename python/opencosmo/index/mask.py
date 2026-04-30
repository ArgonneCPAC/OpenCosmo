from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from opencosmo.index import ChunkedIndex, DataIndex, SimpleIndex

from opencosmo._lib import index as idxlib
from opencosmo.index.unary import get_length


def mask(index: DataIndex, boolean_mask: NDArray[np.bool_]) -> SimpleIndex:
    match index:
        case np.ndarray():
            return __mask_simple(index, boolean_mask)
        case (np.ndarray(), np.ndarray()):
            return __mask_chunked(index, boolean_mask)
        case _:
            raise TypeError(f"Unknown index type {type(index)}")


def __mask_simple(index: SimpleIndex, boolean_mask: NDArray[np.bool_]) -> SimpleIndex:
    if (lm := len(boolean_mask)) > len(index):
        raise ValueError(
            "Boolean mask must be less than or equal to the length of the index itself"
        )

    return index[:lm][boolean_mask]


def __mask_chunked(index: ChunkedIndex, boolean_mask: NDArray[np.bool_]) -> SimpleIndex:
    array = into_array(index)
    return array[boolean_mask]


def into_array(index: DataIndex) -> SimpleIndex:
    if get_length(index) == 0:
        return np.array([], dtype=np.int64)

    match index:
        case np.ndarray():
            return index
        case (np.ndarray(), np.ndarray()):
            if len(index[0]) == 1:
                return np.arange(index[0][0], index[0][0] + index[1][0])

            return idxlib.chunked_into_array(*index)
        case _:
            raise ValueError(f"Expected a DataIndex, got {type(index)}")
