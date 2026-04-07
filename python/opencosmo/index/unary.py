from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opencosmo._lib import index as idx

if TYPE_CHECKING:
    from opencosmo.index import DataIndex

"""
Implementations for unary operations on indices
"""


def get_length(index: DataIndex) -> int:
    match index:
        case np.ndarray():
            return len(index)
        case (np.ndarray(), np.ndarray()):
            return int(np.sum(index[1]))
        case _:
            raise TypeError(f"Invalid index type {type(index)}")


def get_range(index: DataIndex) -> tuple[int, int]:
    match index:
        case np.ndarray():
            return idx.get_simple_range(index)
        case (np.ndarray(), np.ndarray()):
            return idx.get_chunked_range(*index)
        case _:
            raise ValueError(f"Unknown index type {type(index)}")
