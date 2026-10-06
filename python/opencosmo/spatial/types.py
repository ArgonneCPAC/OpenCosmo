from collections.abc import Mapping
from typing import TypeAlias

import h5py
import numpy as np

SpatialIndexData: TypeAlias = (
    h5py.Group | Mapping[str, h5py.Group | h5py.Dataset | np.ndarray]
)
