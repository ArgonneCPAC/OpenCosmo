from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opencosmo._lib import index as idxlib
from opencosmo.index import into_array

if TYPE_CHECKING:
    from opencosmo.index import ChunkedIndex, DataIndex


def reindex_column(index: DataIndex, column: np.ndarray):
    column = column.astype(np.int64)
    return idxlib.reindex_column(into_array(index), column)


def rebuild_by_ranges(index: DataIndex, ranges: ChunkedIndex):
    return idxlib.rebuild_simple_by_ranges(index, *ranges)
