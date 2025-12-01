from functools import singledispatch

import numba as nb
import numpy as np
from numpy.typing import NDArray


@singledispatch
def mask(index: NDArray[np.int_], boolean_mask: NDArray[bool]):
    if (lm := len(boolean_mask)) > len(index):
        raise ValueError(
            "Boolean mask must be less than or equal to the length of the index itself"
        )

    return index[:lm][boolean_mask]


@mask.register
def _(index: tuple, boolean_mask: NDArray[bool]):
    array = into_array(index[0], index[1])
    return array[boolean_mask]


def into_array(index: np.ndarray | tuple):
    match index:
        case np.ndarray():
            return index
        case (np.ndarray(), np.ndarray()):
            return __chunked_into_array(*index)


@nb.njit
def __chunked_into_array(starts: NDArray[np.int_], sizes: NDArray[np.int_]):
    output = np.zeros(np.sum(sizes), dtype=np.int64)
    rs = 0
    for i in range(len(starts)):
        output[rs : rs + sizes[i]] = np.arange(starts[i], starts[i] + sizes[i])
    return output
