from __future__ import annotations

import numpy as np

from opencosmo._lib import index as idxlib

SimpleIndex = np.ndarray
ChunkedIndex = tuple[np.ndarray, np.ndarray]


def take(from_, by):
    match (from_, by):
        case (np.ndarray(), np.ndarray()):
            return __take_simple_from_simple(from_, by)
        case (np.ndarray(), (np.ndarray(), np.ndarray())):
            return idxlib.take_chunked_from_simple(from_, *by)
        case ((np.ndarray(), np.ndarray()), np.ndarray()):
            return __take_simple_from_chunked(from_, by)
        case ((np.ndarray(), np.ndarray()), (np.ndarray(), np.ndarray())):
            print("Chunked from chunked")
            return idxlib.take_chunked_from_chunked(*from_, *by)


def __take_simple_from_chunked(from_: ChunkedIndex, by: SimpleIndex):
    cumulative = np.insert(np.cumsum(from_[1]), 0, 0)[:-1]

    indices_into_chunks = np.argmax(by[:, np.newaxis] < cumulative, axis=1) - 1
    output = by - cumulative[indices_into_chunks] + from_[0][indices_into_chunks]
    return output


def __take_simple_from_simple(from_: np.ndarray, by: np.ndarray):
    return from_[by]
