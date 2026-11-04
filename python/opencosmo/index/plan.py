from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from opencosmo._lib import plan as planlib

from .mask import into_array
from .unary import get_length

# Largest temporary buffer, in bytes, allocated while reading an index from HDF5.
# The output array itself is not counted.
MAX_READ_BUFFER = 8_000_000_000

if TYPE_CHECKING:
    from collections.abc import Mapping

    import h5py

    from opencosmo.index import DataIndex


@dataclass(frozen=True, slots=True, eq=False)
class ReadPlan:
    """
    Ascending HDF5 reads for one index, shared by every column with the same HDF5
    chunk rows and row budget. Run ``k`` reads ``run_sizes[k]`` rows from
    ``run_starts[k]`` and owns the next ``run_segments[k]`` segments, each copying
    ``segment_sizes`` rows from ``segment_local`` in the run to ``segment_output``.
    """

    run_starts: np.ndarray
    run_sizes: np.ndarray
    run_segments: np.ndarray
    segment_output: np.ndarray
    segment_local: np.ndarray
    segment_sizes: np.ndarray
    length: int


def build_plans(
    columns: Mapping[str, h5py.Dataset], index: DataIndex
) -> dict[str, ReadPlan]:
    """
    Build read plans for reading ``index`` from many columns at once.

    A plan depends only on the index, the HDF5 chunk rows, and the row budget
    derived from ``MAX_READ_BUFFER``, so columns that agree on both share one plan.

    Parameters
    ----------
    columns : Mapping[str, h5py.Dataset]
        Column names and the datasets they are read from. All must have the same
        number of rows.
    index : DataIndex
        Rows to read, in output order.

    Returns
    -------
    dict[str, ReadPlan]
        The plan for each column, for use with ``read_with_plan``.
    """
    keys = {name: __plan_key(column) for name, column in columns.items()}
    plans = {
        key: __build_plan(index, *key) for key in sorted(set(keys.values()), key=str)
    }
    return {name: plans[key] for name, key in keys.items()}


def __plan_key(data: h5py.Dataset) -> tuple[int | None, int]:
    # Scatter index arrays are int64, so no temporary costs less than 8 bytes a row
    row_bytes = math.prod(data.shape[1:]) * data.dtype.itemsize
    max_rows = max(MAX_READ_BUFFER // max(row_bytes, np.dtype(np.int64).itemsize), 1)
    chunk_rows = data.chunks[0] if data.chunks is not None else None
    return chunk_rows, max_rows


def __build_plan(index: DataIndex, chunk_rows: int | None, max_rows: int) -> ReadPlan:
    match index:
        case np.ndarray():
            raw = planlib.get_simple_read_plan(index, chunk_rows, max_rows)
        case (np.ndarray(), np.ndarray()):
            raw = planlib.get_chunked_read_plan(*index, chunk_rows, max_rows)
        case _:
            raise ValueError(f"Got invalid index of type {type(index)}")
    return ReadPlan(*raw, length=get_length(index))


def read_with_plan(data: h5py.Dataset, plan: ReadPlan) -> np.ndarray:
    """
    Execute a read plan. A run that fills one contiguous piece of the output is
    read in place; any other run is read into a bounded buffer and scattered into
    the output.
    """
    seg_ends = np.cumsum(plan.run_segments)
    firsts = seg_ends - plan.run_segments
    in_place = (
        (plan.run_segments == 1)
        & (plan.segment_local[firsts] == 0)
        & (plan.segment_sizes[firsts] == plan.run_sizes)
    )
    output = np.empty((plan.length,) + data.shape[1:], dtype=data.dtype)
    buffer: np.ndarray | None = None
    if not in_place.all():
        # Runs larger than the limit always hold a single range read in place
        buffer_rows = int(plan.run_sizes[~in_place].max())
        buffer = np.empty((buffer_rows,) + data.shape[1:], dtype=data.dtype)

    # Iterate the arrays directly; tolist() costs ~36 bytes per run per list
    for start_, size_, first_, last_, direct in zip(
        plan.run_starts, plan.run_sizes, firsts, seg_ends, in_place, strict=True
    ):
        start, size, first, last = int(start_), int(size_), int(first_), int(last_)
        source = np.s_[start : start + size]
        if direct:
            out = int(plan.segment_output[first])
            data.read_direct(output, source, np.s_[out : out + size])
        else:
            assert buffer is not None
            data.read_direct(buffer, source, np.s_[0:size])
            sizes = plan.segment_sizes[first:last]
            output[into_array((plan.segment_output[first:last], sizes))] = buffer[
                into_array((plan.segment_local[first:last], sizes))
            ]
    return output
