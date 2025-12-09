import numpy as np
from numpy.typing import NDArray

from .build import concatenate, empty, from_size, single_chunk
from .get import get_data
from .in_range import n_in_range
from .mask import into_array, mask
from .project import project
from .take import take
from .unary import get_length, get_range

SimpleIndex = NDArray[np.int_]
ChunkedIndex = tuple[NDArray[np.int_], NDArray[np.int_]]
DataIndex = SimpleIndex | ChunkedIndex


__all__ = [
    "DataIndex",
    "SimpleIndex",
    "ChunkedIndex",
    "empty",
    "from_size",
    "single_chunk",
    "concatenate",
    "get_data",
    "get_length",
    "get_range",
    "into_array",
    "mask",
    "n_in_range",
    "project",
    "take",
    "get_length",
    "get_range",
]
