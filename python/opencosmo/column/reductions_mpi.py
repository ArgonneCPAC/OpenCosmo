"""Global MPI reductions for scalar quantities."""

from __future__ import annotations

from functools import partial
from typing import Any

import astropy.units as u  # type: ignore
import numpy as np

from opencosmo.column.column import (
    _max,
    _mean,
    _median,
    _min,
    _quantile,
    _std,
    _sum,
    _var,
)
from opencosmo.mpi import get_comm_world, get_mpi

REDUCTIONS = frozenset([_min, _max, _sum, _mean, _var, _std, _median])


def evaluate_global_reduction(operation: Any, local_data: Any) -> Any:
    """
    Evaluate a reduction operation globally across all MPI ranks.

    If not running under MPI (comm is None), returns the local reduction.
    Otherwise, dispatches to the appropriate MPI collective strategy.

    Parameters
    ----------
    operation
        One of the reduction functions (_min, _max, _sum, _mean, _var, _std,
        _median) or a partial of _quantile.
    local_data
        The local data array (numpy array or astropy Quantity).

    Returns
    -------
    Any
        The global reduction result, with units preserved if the input has them.

    Raises
    ------
    NotImplementedError
        If the operation is not a recognized reduction.
    """
    comm = get_comm_world()
    if comm is None:
        # No MPI — global IS local
        return operation(local_data, None)

    if operation is _min:
        return _allreduce_quantity(operation(local_data, None), get_mpi().MIN, comm)
    if operation is _max:
        return _allreduce_quantity(operation(local_data, None), get_mpi().MAX, comm)
    if operation is _sum:
        return _allreduce_quantity(operation(local_data, None), get_mpi().SUM, comm)
    if operation is _mean:
        # SUM(values) / SUM(counts) — each rank contributes its local sum and count,
        # combined with two SUM allreduces.
        local_sum = (
            local_data.sum() if hasattr(local_data, "sum") else np.sum(local_data)
        )
        local_n = len(local_data)
        gs = _allreduce_quantity(local_sum, get_mpi().SUM, comm)
        gn = comm.allreduce(local_n, op=get_mpi().SUM)
        return gs / gn
    if operation in (_var, _std, _median) or (
        isinstance(operation, partial) and operation.func is _quantile
    ):
        return _gather_and_compute(operation, local_data, comm)
    raise NotImplementedError(f"Global reduction not implemented for {operation}")


def _allreduce_quantity(value: Any, op: Any, comm: Any) -> Any:
    """
    Apply an allreduce to a value, handling astropy Quantities.

    If the value is a Quantity, unwraps it, reduces the .value, then re-attaches
    the unit. Otherwise, reduces directly.
    """
    if isinstance(value, u.Quantity):
        raw = value.value
        unit = value.unit
        global_raw = comm.allreduce(raw, op=op)
        return global_raw * unit
    return comm.allreduce(value, op=op)


def _gather_and_compute(operation: Any, local_data: Any, comm: Any) -> Any:
    """
    Gather all data to rank 0, compute the reduction there, broadcast the result.

    Used for operations like var, std, median, quantile where the local streaming
    formula is complex or where we need the full dataset.

    Parameters
    ----------
    operation
        The reduction operation (_var, _std, _median, or partial(_quantile, q=...))
    local_data
        The local data array.
    comm
        The MPI communicator.

    Returns
    -------
    Any
        The global reduction result, with units preserved.
    """
    # Unwrap Quantity if needed
    is_quantity = isinstance(local_data, u.Quantity)
    if is_quantity:
        unit = local_data.unit
        local_values = np.asarray(local_data.value, dtype=np.float64)
    else:
        local_values = np.asarray(local_data, dtype=np.float64)

    rank = comm.Get_rank()
    lengths = np.array(comm.allgather(len(local_values)), dtype=np.int64)
    total_length = int(np.sum(lengths))

    # Prepare Gatherv receive buffer on rank 0
    offsets = np.zeros(len(lengths), dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)[:-1]
    recv = np.empty(total_length, dtype=np.float64) if rank == 0 else None

    comm.Gatherv(local_values, [recv, lengths, offsets, get_mpi().DOUBLE], root=0)

    # Compute on rank 0
    if rank != 0:
        result = None
    else:
        assert recv is not None
        result = operation(recv, None)

    # Broadcast the result
    result = comm.bcast(result, root=0)

    # Re-attach units if needed
    if is_quantity:
        result = result * unit

    return result
