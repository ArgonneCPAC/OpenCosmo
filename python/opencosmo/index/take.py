from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opencosmo._lib import index as idxlib

if TYPE_CHECKING:
    from opencosmo.index import ChunkedIndex, DataIndex, SimpleIndex


def take(from_: DataIndex, by: DataIndex) -> DataIndex:
    match (from_, by):
        case (np.ndarray(), np.ndarray()):
            return __take_simple_from_simple(from_, by)
        case (np.ndarray(), (np.ndarray(), np.ndarray())):
            return idxlib.take_chunked_from_simple(from_, *by)
        case ((np.ndarray(), np.ndarray()), np.ndarray()):
            return __take_simple_from_chunked(from_, by)
        case ((np.ndarray(), np.ndarray()), (np.ndarray(), np.ndarray())):
            return idxlib.take_chunked_from_chunked(*from_, *by)
        case _:
            raise TypeError(f"Invalid index types: {type(from_)}, {type(by)}")


def __take_simple_from_chunked(from_: ChunkedIndex, by: SimpleIndex) -> SimpleIndex:
    cumulative = np.insert(np.cumsum(from_[1]), 0, 0)[:-1]

    indices_into_chunks = np.argmax(by[:, np.newaxis] < cumulative, axis=1) - 1
    output = by - cumulative[indices_into_chunks] + from_[0][indices_into_chunks]
    return output


def __take_simple_from_simple(from_: SimpleIndex, by: SimpleIndex) -> SimpleIndex:
    return from_[by]
