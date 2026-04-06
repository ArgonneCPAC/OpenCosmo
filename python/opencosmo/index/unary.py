import numpy as np
from numpy.typing import NDArray

from opencosmo._lib import index as idx

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
            return idx.get_simple_range(index)
        case (np.ndarray(), np.ndarray()):
            return idx.get_chunked_range(*index)
        case _:
            raise ValueError(f"Unknown index type {type(index)}")
