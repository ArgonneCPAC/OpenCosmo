from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import h5py
import numpy as np

from .read_plan import plan, scatter
from .unary import get_length

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opencosmo.index import ChunkedIndex, DataIndex

    from .read_plan import ReadPlan

# Upper bound on the scratch buffer for one HDF5 read, in bytes.
READ_BUDGET = 64 << 20


def get_data(data: h5py.Dataset | np.ndarray, index: DataIndex):
    if get_length(index) == 0:
        return np.array([], data.dtype)
    if isinstance(data, h5py.Dataset):
        return __read_planned(data, plan(index, *__layout(data), READ_BUDGET))
    match index:
        case np.ndarray():
            return data[index]
        case (np.ndarray(), np.ndarray()):
            return __get_chunked(data, index)
        case _:
            raise ValueError(f"Got invalid index of type {type(index)}")


def read_columns(
    columns: Mapping[str, h5py.Dataset], index: DataIndex
) -> dict[str, np.ndarray]:
    """
    Read the same rows from several HDF5 columns, in the order given. Columns
    that share a layout share one read plan.
    """
    if get_length(index) == 0:
        return {name: np.array([], column.dtype) for name, column in columns.items()}
    plans: dict[tuple[int, int, int], ReadPlan] = {}
    output = {}
    for name, column in columns.items():
        layout = __layout(column)
        if layout not in plans:
            plans[layout] = plan(index, *layout, READ_BUDGET)
        output[name] = __read_planned(column, plans[layout])
    return output


def __layout(data: h5py.Dataset) -> tuple[int, int, int]:
    """
    The ``(n_rows, chunk_rows, row_bytes)`` a read plan depends on. Unchunked
    columns are planned as if chunked to the read budget.
    """
    row_bytes = max(1, data.dtype.itemsize * int(np.prod(data.shape[1:])))
    chunk_rows = (
        data.chunks[0] if data.chunks is not None else max(1, READ_BUDGET // row_bytes)
    )
    return (data.shape[0], chunk_rows, row_bytes)


def __read_planned(data: h5py.Dataset, read_plan: ReadPlan) -> np.ndarray:
    """
    Read one window at a time, so each storage chunk is read and decompressed
    once. A window that selects every row in its span is read straight into the
    output; others go through a scratch buffer of at most ``READ_BUDGET`` bytes
    (or one chunk, if larger).
    """
    row_shape = data.shape[1:]
    windows = read_plan.windows

    out = np.empty((read_plan.length,) + row_shape, data.dtype)
    n_rows = windows.row_hi - windows.row_lo
    dense = windows.out_hi - windows.out_lo == n_rows
    scratch = np.empty((int(n_rows[~dense].max(initial=0)),) + row_shape, data.dtype)
    for w in range(len(n_rows)):
        lo, hi = int(windows.row_lo[w]), int(windows.row_hi[w])
        if dense[w]:
            dest = int(windows.out_lo[w])
            data.read_direct(out, np.s_[lo:hi], np.s_[dest : dest + hi - lo])
            continue
        data.read_direct(scratch, np.s_[lo:hi], np.s_[0 : hi - lo])
        scatter(read_plan, w, scratch[: hi - lo], out)

    return out if read_plan.gather is None else out[read_plan.gather]


def __get_chunked(data: np.ndarray, index: ChunkedIndex) -> np.ndarray:
    starts, sizes = index
    unit = data.unit if isinstance(data, u.Quantity) else None

    storage = np.zeros((np.sum(sizes),) + data.shape[1:], dtype=data.dtype)
    running_index = 0
    for start, size in zip(starts, sizes):
        storage[running_index : running_index + size] = data[start : start + size]
        running_index += int(size)

    if unit is not None:
        storage *= unit
    return storage
