"""
Planning for chunk-aware reads.

Compressed HDF5 data can only be decompressed a whole chunk at a time. A read is
planned as windows of consecutive storage chunks, so each chunk is fetched and
decompressed exactly once and scratch memory stays within a fixed budget. A plan
holds a handful of integers per window and refers back into the index, so its
size does not grow with the index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from opencosmo._lib import plan as planlib

from .mask import into_array
from .unary import get_length

if TYPE_CHECKING:
    from opencosmo.index import DataIndex, IndexArray


@dataclass(frozen=True)
class ReadWindows:
    """
    Window ``w`` holds rows ``[row_lo[w], row_hi[w])``, a run of whole storage
    chunks that are all touched by the index. It covers index positions
    ``[pos_lo[w], pos_hi[w])``: elements of a simple index, or segments of a
    chunked index, the first and last of which may extend past the window. Its
    first selected row lands at ``out_lo[w]`` in the output.
    """

    row_lo: IndexArray
    row_hi: IndexArray
    pos_lo: IndexArray
    pos_hi: IndexArray
    out_lo: IndexArray


@dataclass(frozen=True)
class ReadPlan:
    """
    ``index`` is sorted and free of repeats, and is what ``windows`` refers to.
    When the caller's index was not, ``gather`` restores its order and repeats:
    the caller's result is ``output[gather]``.
    """

    index: DataIndex
    windows: ReadWindows
    length: int
    gather: IndexArray | None


def plan(
    index: DataIndex, n_rows: int, chunk_rows: int, row_bytes: int, budget: int
) -> ReadPlan:
    """
    Plan a read of ``index`` from a column of ``n_rows`` rows stored in chunks of
    ``chunk_rows``. A window holds at most ``budget`` bytes, but never less than
    one chunk.
    """
    if chunk_rows <= 0 or row_bytes <= 0 or budget <= 0:
        raise ValueError("chunk_rows, row_bytes, and budget must be positive")
    max_chunks = max(1, budget // (chunk_rows * row_bytes))

    if (windows := __plan_windows(index, n_rows, chunk_rows, max_chunks)) is not None:
        return ReadPlan(index, windows, get_length(index), None)

    # Rare: the caller asked for rows out of order or repeated. Read each row once,
    # in order, and rearrange afterwards.
    rows, gather = np.unique(into_array(index), return_inverse=True)
    rows = rows.astype(np.int64)
    windows = __plan_windows(rows, n_rows, chunk_rows, max_chunks)
    if windows is None:
        raise RuntimeError("Sorted unique rows were rejected by the read planner")
    return ReadPlan(rows, windows, len(rows), gather.astype(np.int64))


def __plan_windows(
    index: DataIndex, n_rows: int, chunk_rows: int, max_chunks: int
) -> ReadWindows | None:
    match index:
        case np.ndarray():
            windows = planlib.plan_windows_simple(
                index.astype(np.int64, copy=False), n_rows, chunk_rows, max_chunks
            )
        case (np.ndarray(), np.ndarray()):
            windows = planlib.plan_windows_chunked(
                index[0].astype(np.int64, copy=False),
                index[1].astype(np.int64, copy=False),
                n_rows,
                chunk_rows,
                max_chunks,
            )
        case _:
            raise TypeError(f"Invalid index type {type(index)}")
    return None if windows is None else ReadWindows(*windows)


def scatter(read_plan: ReadPlan, w: int, window_data: np.ndarray, out: np.ndarray):
    """
    Copy the selected rows of window ``w``, whose rows are ``window_data``, into
    ``out``.
    """
    windows = read_plan.windows
    row_lo = int(windows.row_lo[w])
    pos_lo, pos_hi = int(windows.pos_lo[w]), int(windows.pos_hi[w])
    match read_plan.index:
        case np.ndarray():
            out[pos_lo:pos_hi] = window_data[read_plan.index[pos_lo:pos_hi] - row_lo]
        case (starts, sizes):
            row_hi = row_lo + len(window_data)
            dest = int(windows.out_lo[w])
            for start, size in zip(starts[pos_lo:pos_hi], sizes[pos_lo:pos_hi]):
                lo, hi = max(int(start), row_lo), min(int(start + size), row_hi)
                if hi <= lo:
                    continue
                out[dest : dest + hi - lo] = window_data[lo - row_lo : hi - row_lo]
                dest += hi - lo
