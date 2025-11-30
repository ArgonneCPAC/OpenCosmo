import numpy as np
from numpy.typing import NDArray

from .build import from_size
from .chunked import ChunkedIndex
from .protocols import DataIndex
from .simple import SimpleIndex

__all__ = ["DataIndex", "SimpleIndex", "ChunkedIndex", "from_size"]
