from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opencosmo._lib import index as idxlib

from . import into_array

if TYPE_CHECKING:
    from opencosmo.index import ChunkedIndex, DataIndex, SimpleIndex


def project(source: DataIndex, other: DataIndex) -> DataIndex:
    match (source, other):
        case (tuple(), np.ndarray()):
            return __project_simple_on_chunked(source, other)
        case (tuple(), tuple()):
            return __project_chunked_on_chunked(source, other)
        case (np.ndarray(), np.ndarray()):
            return __project_simple_on_simple(source, other)
        case (np.ndarray(), tuple()):
            return __project_chunked_on_simple(source, other)
        case _:
            raise TypeError(f"Invalid index types: {type(source)}, {type(other)}")


def __project_simple_on_simple(source: SimpleIndex, other: SimpleIndex) -> SimpleIndex:
    isin = np.isin(source, other)
    return np.where(isin)[0]


def __project_chunked_on_simple(
    source: SimpleIndex, other: ChunkedIndex
) -> SimpleIndex:
    if len(other[0]) == 0:
        return np.array([], dtype=np.int64)
    return idxlib.project_chunked_on_simple(source, *other)


def __project_simple_on_chunked(source: ChunkedIndex, other: SimpleIndex) -> DataIndex:
    return project(into_array(source), other)


def __project_chunked_on_chunked(
    source: ChunkedIndex, other: ChunkedIndex
) -> ChunkedIndex:
    source_ends = source[0] + source[1]
    other_ends = other[0] + other[1]

    clipped_starts = np.clip(other[0][:, np.newaxis], a_min=source[0], a_max=None)
    clipped_ends = np.clip(other_ends[:, np.newaxis], a_min=None, a_max=source_ends)

    rs = np.cumsum(source[1])
    rs = np.insert(rs, 0, 0)[:-1]

    starts = clipped_starts - source[0] + rs
    ends = clipped_ends - source[0] + rs
    sizes = ends - starts

    starts = starts.flatten()
    sizes = sizes.flatten()

    mask = sizes > 0
    return (starts[mask], sizes[mask])
