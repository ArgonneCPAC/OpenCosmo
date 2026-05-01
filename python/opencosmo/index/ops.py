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
    match index:
        case np.ndarray():
            return idxlib.rebuild_simple_by_ranges(index, *ranges)
        case (np.ndarray(), np.ndarray()):
            return idxlib.rebuild_chunked_by_ranges(*index, *ranges)


def offset(index: DataIndex, offset_amount: int):
    if isinstance(index, np.ndarray):
        return index + offset_amount
    return (index[0] + offset_amount, index[1])
