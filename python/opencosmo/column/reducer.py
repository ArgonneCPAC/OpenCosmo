from __future__ import annotations

from functools import partial
from typing import Any, Callable, Protocol

import astropy.units as u  # type: ignore
import numpy as np

from opencosmo.mpi import get_comm_world, get_mpi


class Reducer(Protocol):
    """
    Protocol for reducing local data to a final result. Different implementations
    handle different partition axes: local (single rank, single dataset),
    MPI-global (across ranks of a dataset), or lightcone-scoped (across steps,
    then optionally across MPI ranks).
    """

    def reduce(self, operation: Callable, local_data: Any) -> Any:
        """
        Apply a reduction operation to local data and return the final result.

        Parameters
        ----------
        operation : Callable
            A reduction function like _min, _max, _mean, etc., or a partial(_quantile, q=...).
        local_data : Any
            The local data (numpy array or astropy Quantity).

        Returns
        -------
        Any
            The reduced result, with units preserved if the input has them.
        """
        ...


class LocalReducer:
    """
    Reducer that computes reductions locally without any cross-process/cross-step combination.
    """

    def reduce(self, operation: Callable, local_data: Any) -> Any:
        """Apply the reduction operation to local data."""
        return operation(local_data, None)


class MpiReducer:
    """
    Reducer that combines local reductions across MPI ranks.
    """

    def __init__(self, comm: Any):
        """
        Parameters
        ----------
        comm : MPI.Comm or None
            The MPI communicator. If None, behaves like LocalReducer.
        """
        self.comm = comm

    def reduce(self, operation: Callable, local_data: Any) -> Any:
        """
        Evaluate a reduction operation globally across all MPI ranks.

        If comm is None, returns the local reduction.
        Otherwise, dispatches to the appropriate MPI collective strategy.

        Parameters
        ----------
        operation : Callable
            One of the reduction functions (_min, _max, _sum, _mean, _var, _std,
            _median) or a partial of _quantile.
        local_data : Any
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

        if self.comm is None:
            return operation(local_data, None)

        local_n = len(local_data)
        global_n = self.comm.allreduce(local_n, op=get_mpi().SUM)
        if global_n == 0:
            raise ValueError(
                "Cannot compute reduction on a globally empty dataset (all ranks have 0 rows)."
            )

        local_unit = local_data.unit if isinstance(local_data, u.Quantity) else None
        all_units = self.comm.allgather(local_unit)
        unit = next((u for u in all_units if u is not None), None)

        if operation is _min or operation is _max:
            identity = np.inf if operation is _min else -np.inf
            mpi_op = get_mpi().MIN if operation is _min else get_mpi().MAX
            if local_n > 0:
                local_value = operation(local_data, None)
            else:
                local_value = identity * unit if unit is not None else identity
            return self.comm.allreduce(local_value, op=mpi_op)
        if operation is _sum:
            local_value = (
                operation(local_data, None)
                if local_n > 0
                else (0.0 * unit if unit is not None else 0.0)
            )
            return self.comm.allreduce(local_value, op=get_mpi().SUM)
        if operation is _mean:
            local_sum = (
                np.sum(local_data)
                if local_n > 0
                else (0.0 * unit if unit is not None else 0.0)
            )
            gs = self.comm.allreduce(local_sum, op=get_mpi().SUM)
            return gs / global_n
        if operation in (_var, _std, _median) or (
            isinstance(operation, partial) and operation.func is _quantile
        ):
            return self._gather_and_compute(operation, local_data, unit)
        raise NotImplementedError(f"Global reduction not implemented for {operation}")

    def _gather_and_compute(self, operation: Any, local_data: Any, unit: Any) -> Any:
        """
        Gather all data to rank 0, compute the reduction there, broadcast the result.

        Used for operations like var, std, median, quantile where the local streaming
        formula is complex or where we need the full dataset.

        Parameters
        ----------
        operation : Callable
            The reduction operation (_var, _std, _median, or partial(_quantile, q=...))
        local_data : Any
            The local data array.
        unit : u.Unit or None
            The unit to re-attach if needed.

        Returns
        -------
        Any
            The global reduction result, with units preserved.
        """
        local_values = np.asarray(
            local_data.value if isinstance(local_data, u.Quantity) else local_data,
            dtype=np.float64,
        )

        rank = self.comm.Get_rank()
        lengths = np.array(self.comm.allgather(len(local_values)), dtype=np.int64)
        total_length = int(np.sum(lengths))

        # Prepare Gatherv receive buffer on rank 0
        offsets = np.zeros(len(lengths), dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)[:-1]
        recv = np.empty(total_length, dtype=np.float64) if rank == 0 else None

        self.comm.Gatherv(
            local_values, [recv, lengths, offsets, get_mpi().DOUBLE], root=0
        )

        # Compute on rank 0
        if rank != 0:
            result = None
        else:
            assert recv is not None
            result = operation(recv, None)

        # Broadcast the result
        result = self.comm.bcast(result, root=0)

        # Re-attach units if needed
        if unit is not None:
            result = result * unit

        return result


def default_reducer() -> Reducer:
    """
    Get the default reducer for the current context.
    If MPI is available, returns MpiReducer(comm). Otherwise, returns LocalReducer.
    """
    comm = get_comm_world()
    return MpiReducer(comm) if comm is not None else LocalReducer()
