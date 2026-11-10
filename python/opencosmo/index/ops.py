from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opencosmo._lib import index as idxlib
from opencosmo.index import coalesce_chunks, into_array

if TYPE_CHECKING:
    from opencosmo.index import ChunkedIndex, DataIndex, SimpleIndex


def reindex_column(index: DataIndex, column: np.ndarray) -> SimpleIndex:
    column = column.astype(np.int64)
    return idxlib.reindex_column(into_array(index), column)


def rebuild_by_ranges(index: DataIndex, ranges: ChunkedIndex):
    match index:
        case np.ndarray():
            return idxlib.rebuild_simple_by_ranges(index, *ranges)
        case (np.ndarray(), np.ndarray()):
            return idxlib.rebuild_chunked_by_ranges(*index, *ranges)


def trim(index: DataIndex, max: int):
    if isinstance(index, np.ndarray):
        return index[index < max]
    ends = index[0] + index[1]
    ends = np.clip(ends, a_min=None, a_max=max)

    valid = ends > index[0]
    new_sizes = ends - index[0]
    return index[0][valid], new_sizes[valid]


def offset(index: DataIndex, offset_amount: int):
    if isinstance(index, np.ndarray):
        return index + offset_amount
    return (index[0] + offset_amount, index[1])


def combine(a: DataIndex, b: DataIndex) -> DataIndex:
    """
    Combine two indices into one covering the union of their rows.

    - Chunked + Chunked -> Chunked with overlapping/adjacent ranges merged.
    - Chunked + Simple  -> Simple (deduplicated).
    - Simple  + Simple  -> Simple (deduplicated).
    """
    a_is_chunked = isinstance(a, tuple)
    b_is_chunked = isinstance(b, tuple)
    if a_is_chunked and b_is_chunked:
        return __combine_chunked_chunked(a, b)
    if a_is_chunked or b_is_chunked:
        # At least one chunked, one simple -> materialize to simple, deduplicate
        arr = np.union1d(into_array(a), into_array(b))
        return arr.astype(np.int64, copy=False)
    # Simple + Simple
    arr = np.union1d(a, b)
    return arr.astype(np.int64, copy=False)


def __combine_chunked_chunked(a: ChunkedIndex, b: ChunkedIndex) -> ChunkedIndex:
    starts = np.concatenate((a[0], b[0]))
    sizes = np.concatenate((a[1], b[1]))
    if len(starts) == 0:
        return (starts.astype(np.int64, copy=False), sizes.astype(np.int64, copy=False))
    order = np.argsort(starts, kind="stable")
    starts = starts[order]
    sizes = sizes[order]
    return coalesce_chunks(starts, sizes)


def sort(index: DataIndex):
    if isinstance(index, np.ndarray):
        return np.sort(index)

    sort_order = np.argsort(index[0])
    return (index[0][sort_order], index[1][sort_order])
