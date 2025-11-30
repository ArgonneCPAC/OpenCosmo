import numpy as np


def from_size(size: int):
    return (np.atleast_1d(0), np.atleast_1d(size))


def single_chunk(start: int, size: int):
    return (np.atleast_1d(start), np.atleast_1d(size))
