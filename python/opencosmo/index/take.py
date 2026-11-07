from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard

import numpy as np

from opencosmo._lib import index as idxlib

if TYPE_CHECKING:
    from opencosmo.index import ChunkedIndex, DataIndex, SimpleIndex


def __is_simple(index: DataIndex) -> TypeGuard[SimpleIndex]:
    return isinstance(index, np.ndarray)


def __is_chunked(index: DataIndex) -> TypeGuard[ChunkedIndex]:
    return (
        isinstance(index, (tuple, list))
        and len(index) == 2
        and all(isinstance(part, np.ndarray) for part in index)
    )


def take(from_: DataIndex, by: DataIndex) -> DataIndex:
    if __is_simple(from_) and __is_simple(by):
        return __take_simple_from_simple(from_, by)
    if __is_simple(from_) and __is_chunked(by):
        return idxlib.take_chunked_from_simple(from_, *by)
    if __is_chunked(from_) and __is_simple(by):
        return __take_simple_from_chunked(from_, by)
    if __is_chunked(from_) and __is_chunked(by):
        return idxlib.take_chunked_from_chunked(*from_, *by)
    raise TypeError(f"Invalid index types: {type(from_)}, {type(by)}")


def __take_simple_from_chunked(from_: ChunkedIndex, by: SimpleIndex) -> SimpleIndex:
    cumulative = np.insert(np.cumsum(from_[1]), 0, 0)[:-1]

    indices_into_chunks = np.argmax(by[:, np.newaxis] < cumulative, axis=1) - 1
    output = by - cumulative[indices_into_chunks] + from_[0][indices_into_chunks]
    return output


def __take_simple_from_simple(from_: SimpleIndex, by: SimpleIndex) -> SimpleIndex:
    return from_[by]
