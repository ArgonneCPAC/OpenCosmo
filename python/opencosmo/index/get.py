from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from opencosmo._lib import index as idxlib

from .mask import into_array
from .unary import get_length

# Largest temporary buffer, in bytes, allocated while reading an index from HDF5.
# The output array itself is not counted.
MAX_READ_BUFFER = 8_000_000_000

if TYPE_CHECKING:
    import h5py

    from opencosmo.index import DataIndex


def get_data(data: h5py.Dataset | np.ndarray, index: DataIndex):
    if get_length(index) == 0:
        return np.array([], data.dtype)
    elif isinstance(data, np.ndarray):
        return data[into_array(index)]

    chunk_size = data.chunks[0] if data.chunks is not None else None
    max_rows = __max_buffer_rows(data)
    match index:
        case np.ndarray():
            plan = idxlib.get_simple_read_plan(index, chunk_size, max_rows)
        case (np.ndarray(), np.ndarray()):
            plan = idxlib.get_chunked_read_plan(*index, chunk_size, max_rows)
        case _:
            raise ValueError(f"Got invalid index of type {type(index)}")
    return __read_plan(data, plan, get_length(index))


def __max_buffer_rows(data: h5py.Dataset) -> int:
    # Scatter index arrays are int64, so no temporary costs less than 8 bytes a row
    row_bytes = math.prod(data.shape[1:]) * data.dtype.itemsize
    return max(MAX_READ_BUFFER // max(row_bytes, np.dtype(np.int64).itemsize), 1)


def __read_plan(
    data: h5py.Dataset,
    plan: tuple[np.ndarray, ...],
    length: int,
) -> np.ndarray:
    """
    Execute a read plan from the Rust kernel. A run that fills one contiguous
    piece of the output is read in place; any other run is read into a bounded
    buffer and scattered into the output.
    """
    run_starts, run_sizes, run_segments, seg_output, seg_local, seg_sizes = plan
    seg_ends = np.cumsum(run_segments)
    firsts = seg_ends - run_segments
    in_place = (
        (run_segments == 1)
        & (seg_local[firsts] == 0)
        & (seg_sizes[firsts] == run_sizes)
    )
    output = np.empty((length,) + data.shape[1:], dtype=data.dtype)
    buffer: np.ndarray | None = None
    if not in_place.all():
        # Runs larger than the limit always hold a single range read in place
        buffer_rows = int(run_sizes[~in_place].max())
        buffer = np.empty((buffer_rows,) + data.shape[1:], dtype=data.dtype)

    for start_, size_, first_, last_, direct in zip(
        run_starts, run_sizes, firsts, seg_ends, in_place, strict=True
    ):
        start, size, first, last = int(start_), int(size_), int(first_), int(last_)
        source = np.s_[start : start + size]
        if direct:
            out = int(seg_output[first])
            data.read_direct(output, source, np.s_[out : out + size])
        else:
            assert buffer is not None
            data.read_direct(buffer, source, np.s_[0:size])
            sizes = seg_sizes[first:last]
            output[into_array((seg_output[first:last], sizes))] = buffer[
                into_array((seg_local[first:last], sizes))
            ]
    return output
