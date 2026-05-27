from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from opencosmo.io import writers as iow


from opencosmo.mpi import MPI


def apply_updaters(
    writers: dict[str, dict[str, iow.ColumnWriter]], comm: Optional[MPI.Comm]
):
    if "data_linked" in writers:
        apply_linked_data_updaters(writers["data_linked"], comm)

    if "index" in writers:
        apply_index_updaters(writers["index"], comm)
    return writers


def apply_index_updaters(columns, comm):
    for colname, column in columns.items():
        column.updater = partial(sum_updater, comm=comm)


def sum_updater(data: np.ndarray, comm: Optional["MPI.Comm"] = None):
    if comm is not None and comm.Get_size():
        recvbuf = np.zeros_like(data)
        comm.Allreduce(data, recvbuf, MPI.SUM)
        return recvbuf
    return data


def apply_linked_data_updaters(columns, comm):
    starts = {name: col for name, col in columns.items() if "start" in name}
    sizes = {name: col for name, col in columns.items() if "size" in name}
    idxs = {name: col for name, col in columns.items() if "idx" in name}
    for name, start in starts.items():
        link_type = name.rsplit("_", maxsplit=1)[0]
        size = sizes[f"{link_type}_size"]
        start.source = size.source
        start.updater = partial(start_link_updater, comm=comm)
    for idx in idxs.values():
        idx.updater = partial(idx_link_updater, comm=comm)


def idx_link_updater(input: np.ndarray, comm) -> np.ndarray:
    output = np.full(len(input), -1)
    good = input >= 0
    if comm is not None:
        all_good = comm.allgather(sum(good))
        offsets = np.insert(np.cumsum(all_good), 0, 0)[:-1]
        offset = offsets[comm.Get_rank()]
    else:
        offset = 0

    output[good] = np.arange(sum(good)) + offset
    return output


def start_link_updater(sizes: np.ndarray, comm) -> np.ndarray:
    cumulative_sizes = np.cumsum(sizes)
    new_starts = np.insert(cumulative_sizes, 0, 0)
    if comm is not None:
        all_total_size = comm.allgather(np.sum(sizes))
        offsets = np.insert(np.cumsum(all_total_size), 0, 0)[:-1]
        offset = offsets[comm.Get_rank()]
    else:
        offset = 0

    new_starts = new_starts[:-1] + offset
    return new_starts
