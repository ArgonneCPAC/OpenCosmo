"""
A column plugin is capable of updating the values of column dynamically when it is instantiated. It is a simple function where the first argument is the name of the column. Additional arguments can take:

1. Additional columns (by name)
2. The dataset index "index"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opencosmo.index import DataIndex


import numpy as np

from opencosmo.index import into_array


def top_host_idx_updater(top_host_idx: np.ndarray, index: DataIndex):
    top_host_idx = top_host_idx.astype(np.int64)
    index_array = into_array(index)

    index_array_sort_idx = np.argsort(index_array)
    index_sorted = index_array[index_array_sort_idx]

    positions = np.searchsorted(index_sorted, top_host_idx)
    valid = (positions < len(index_sorted)) & (index_sorted[positions] == top_host_idx)

    new_top_host_idx = np.full_like(top_host_idx, -1)
    new_top_host_idx[valid] = index_sorted[positions[valid]]
    return valid
