from collections.abc import Mapping

import h5py
import numpy as np

type SpatialIndexData = (
    h5py.Group | Mapping[str, h5py.Group | h5py.Dataset | np.ndarray]
)
