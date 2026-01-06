import numba as nb
import numpy as np
from numpy.typing import NDArray

"""
Implementations for unary operations on indices
"""

SimpleIndex = NDArray[np.int_]
ChunkedIndex = tuple[NDArray[np.int_], NDArray[np.int_]]


def get_length(index: SimpleIndex | ChunkedIndex):
    match index:
        case np.ndarray():
            return len(index)
        case (np.ndarray(), np.ndarray()):
            return int(np.sum(index[1]))
        case _:
            raise TypeError(f"Invalid index type {type(index)}")


def get_range(index: SimpleIndex | ChunkedIndex):
    match index:
        case np.ndarray():
            return __get_simple_range(index)
        case (np.ndarray(), np.ndarray()):
            return __get_chunked_range(*index)
        case _:
            raise ValueError(f"Unknown index type {type(index)}")


@nb.njit
def __get_simple_range(index: SimpleIndex):
    if len(index) == 0:
        return (0, 0)

    min = index[0]
    max = index[0]
    for val in index[1:]:
        if val < min:
            min = val
        if val > max:
            max = val
    return (min, max)


@nb.njit
def __get_chunked_range(starts: NDArray[np.int_], sizes: NDArray[np.int_]):
    if len(starts) == 0:
        return (0, 0)
    min = starts[0]
    max = min + sizes[0]
    for i in range(1, len(starts)):
        start = starts[i]
        end = start + sizes[i]
        if start < min:
            min = start
        if end > max:
            max = end
    return (min, max)
