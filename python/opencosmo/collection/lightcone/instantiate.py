from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from opencosmo.dataset.graph import evaluate_producers

if TYPE_CHECKING:
    from astropy.table import Table  # type: ignore

    from opencosmo.collection.lightcone.scope import LightconeScope


def evaluate_scope(
    scope: LightconeScope,
    vstacked: Table,
    *,
    unpack: bool,
) -> tuple[Table, dict[str, Any] | None]:
    """
    Evaluate scope producers against the vstacked per-child table.

    Returns ``(table, scalars)``. A given scope is guaranteed (by
    ``Lightcone.select`` and ``Lightcone.with_new_columns``) to be either
    all-vector or all-scalar at its top level:

    - All-vector (or empty) scope: ``scalars`` is ``None``; scope outputs are
      written into ``vstacked`` in place.
    - All-scalar scope: ``scalars`` is a dict of {name: value} with the same
      unpack rules ``Dataset.get_data`` applies (length-1 ndarrays and
      shape-(1,) Quantities collapse to bare scalars when ``unpack`` is
      true). ``vstacked`` is returned unchanged so the caller can short-
      circuit to the scalar return path.
    """
    if not scope.names():
        return vstacked, None

    produced = evaluate_producers(
        list(scope.producers),
        {name: vstacked[name] for name in vstacked.colnames},
    )

    scalar_names = scope.scalar_names()
    derived_names = {
        name
        for producer in scope.derived_producers
        for name in (producer.produces or set())
    }
    scalar_outputs = {
        name: value for name, value in produced.items() if name in scalar_names
    }
    vector_outputs = {
        name: value
        for name, value in produced.items()
        if name in derived_names and name not in scalar_names
    }

    if scalar_outputs and vector_outputs:
        raise AssertionError(
            "Lightcone scope must be all-scalar or all-vector at its top level; "
            "Lightcone.select / with_new_columns should have rejected this."
        )

    if vector_outputs:
        for name, values in vector_outputs.items():
            vstacked[name] = values
        return vstacked, None

    if unpack:
        scalar_outputs = {
            name: _unpack_scalar(value) for name, value in scalar_outputs.items()
        }
    return vstacked, scalar_outputs


def _unpack_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == 1:
        return value[0]
    if hasattr(value, "shape") and value.shape == (1,):
        return value[0]
    return value
