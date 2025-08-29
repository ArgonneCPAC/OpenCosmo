from typing import TYPE_CHECKING

import numpy as np

from opencosmo.index import SimpleIndex

if TYPE_CHECKING:
    from opencosmo.dataset.state import DatasetState


def make_sorted_index(state: "DatasetState", column: np.ndarray):
    sorted = np.argsort(column)
    existing_index = state.index.into_array()
    new_index = SimpleIndex(existing_index[sorted])
    return new_index
