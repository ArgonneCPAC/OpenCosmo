"""
Simulation collection operations on SimulationState.

Row operations on a matched collection apply only to the active source and
defer realigning the other members (see ``SimulationState.children``). All
other operations map over the members and keep the current alignment, since
matching only reads raw row indices.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from opencosmo.collection.lightcone import operations as lcops
from opencosmo.collection.lightcone.state import LightconeState
from opencosmo.collection.structure import operations as scops
from opencosmo.column.select import do_multi_dataset_drops, do_multi_dataset_selections
from opencosmo.dataset import operations as dsops
from opencosmo.dataset.state import DatasetState

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import astropy.units as u
    import numpy as np

    from opencosmo.collection.simulation.state import SimulationChild, SimulationState
    from opencosmo.column.column import ColumnMask, ConstructedColumn
    from opencosmo.spatial.protocols import Region

    Selector = Callable[["ChildOps"], Callable[..., Any]]


@dataclass(frozen=True)
class ChildOps:
    """The operations a simulation collection maps over its members."""

    filter: Callable[..., SimulationChild]
    bound: Callable[..., SimulationChild]
    sort_by: Callable[..., SimulationChild]
    take: Callable[..., SimulationChild]
    take_range: Callable[..., SimulationChild]
    select: Callable[..., SimulationChild]
    drop: Callable[..., SimulationChild]
    with_new_columns: Callable[..., SimulationChild]
    with_units: Callable[..., SimulationChild]
    evaluate: Callable[..., Any]


__DATASET_OPS = ChildOps(
    filter=dsops.filter,
    bound=dsops.bound,
    sort_by=dsops.sort_by,
    take=dsops.take,
    take_range=dsops.take_range,
    select=dsops.select,
    drop=dsops.drop,
    with_new_columns=dsops.with_new_columns,
    with_units=dsops.with_units,
    evaluate=dsops.evaluate,
)

__LIGHTCONE_OPS = ChildOps(
    filter=lcops.filter,
    bound=lcops.bound,
    sort_by=lcops.sort_by,
    take=lcops.take,
    take_range=lcops.take_range,
    select=lcops.select,
    drop=lcops.drop,
    with_new_columns=lcops.with_new_columns,
    with_units=lcops.with_units,
    evaluate=lcops.evaluate,
)

__STRUCTURE_OPS = ChildOps(
    filter=scops.filter,
    bound=scops.bound,
    sort_by=scops.sort_by,
    take=scops.take,
    take_range=scops.take_range,
    select=scops.select,
    drop=scops.drop,
    with_new_columns=scops.with_new_columns,
    with_units=scops.with_units,
    evaluate=scops.evaluate,
)


def __child_ops(child: SimulationChild) -> ChildOps:
    if isinstance(child, DatasetState):
        return __DATASET_OPS
    if isinstance(child, LightconeState):
        return __LIGHTCONE_OPS
    return __STRUCTURE_OPS


def map_children(
    state: SimulationState,
    op: Selector,
    *args: Any,
    datasets: str | Iterable[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Apply an operation to the requested members, in member order. A kwarg whose
    value is a dict keyed by exactly the requested names is split per member.
    """
    requested = __requested(state, datasets)
    regular_kwargs = {}
    mapped_kwargs = {}
    for name, value in kwargs.items():
        if isinstance(value, dict) and set(value.keys()) == requested:
            mapped_kwargs[name] = value
        else:
            regular_kwargs[name] = value

    output = {}
    for name, child in state.members.items():
        if name not in requested:
            continue
        child_kwargs = {key: value[name] for key, value in mapped_kwargs.items()}
        output[name] = op(__child_ops(child))(
            child, *args, **regular_kwargs, **child_kwargs
        )
    return output


# Row operations


def filter(
    state: SimulationState,
    *masks: ColumnMask,
    datasets: str | Iterable[str] | None = None,
    **kwargs: Any,
) -> SimulationState:
    return __row_operation(
        state, "filter", lambda ops: ops.filter, datasets, *masks, **kwargs
    )


def bound(
    state: SimulationState, region: Region, select_by: str | None = None
) -> SimulationState:
    return __row_operation(
        state, "bound", lambda ops: ops.bound, None, region, select_by
    )


def sort_by(state: SimulationState, column: str, invert: bool) -> SimulationState:
    return __row_operation(
        state, "sort_by", lambda ops: ops.sort_by, None, column, invert
    )


def take(state: SimulationState, n: int, at: str, mode: str) -> SimulationState:
    return __row_operation(state, "take", lambda ops: ops.take, None, n, at, mode)


def take_range(
    state: SimulationState, start: int, end: int, mode: str
) -> SimulationState:
    return __row_operation(
        state, "take_range", lambda ops: ops.take_range, None, start, end, mode
    )


# Column operations


def select(
    state: SimulationState, *args: str | Iterable[str], **kwargs: Any
) -> SimulationState:
    datasets = __multi_dataset_members(state)
    if datasets is None:
        return __map(state, lambda ops: ops.select, *args, **kwargs)
    return replace(state, members=do_multi_dataset_selections(datasets, args, kwargs))


def drop(
    state: SimulationState, *args: str | Iterable[str], **kwargs: Any
) -> SimulationState:
    datasets = __multi_dataset_members(state)
    if datasets is None:
        return __map(state, lambda ops: ops.drop, *args, **kwargs)

    output = do_multi_dataset_drops(datasets, args)
    for name, columns in kwargs.items():
        if name not in output:
            raise ValueError(f"Dataset {name} not found in collection.")
        output[name] = __child_ops(output[name]).drop(output[name], columns)
    return replace(state, members=output)


def with_new_columns(
    state: SimulationState,
    *args: Any,
    datasets: str | Iterable[str] | None = None,
    descriptions: str | dict[str, str] = {},
    allow_overwrite: bool = False,
    **new_columns: ConstructedColumn | np.ndarray,
) -> SimulationState:
    return __map(
        state,
        lambda ops: ops.with_new_columns,
        *args,
        datasets=datasets,
        descriptions=descriptions,
        allow_overwrite=allow_overwrite,
        **new_columns,
    )


def with_units(
    state: SimulationState,
    convention: str | None,
    conversions: dict[u.Unit, u.Unit],
    **columns: u.Unit,
) -> SimulationState:
    return __map(
        state,
        lambda ops: ops.with_units,
        convention,
        conversions=conversions,
        **columns,
    )


def __row_operation(
    state: SimulationState,
    method: str,
    op: Selector,
    datasets: str | Iterable[str] | None,
    *args: Any,
    **kwargs: Any,
) -> SimulationState:
    source = state.match_source
    if source is None:
        return __map(state, op, *args, datasets=datasets, **kwargs)

    # datasets=None means "all datasets", which for a matched collection
    # resolves to the active source. Any explicit request must name that
    # source and nothing else.
    if isinstance(datasets, str):
        datasets = [datasets]
    if datasets is not None and tuple(datasets) != (source,):
        raise ValueError(
            f"When working with a matched collection, {method} can only be called on the active source. Got datasets = {datasets}"
        )
    new_source = op(__child_ops(state.members[source]))(
        state.members[source], *args, **kwargs
    )
    return replace(state, members={**state.members, source: new_source}, aligned=False)


def __map(
    state: SimulationState, op: Selector, *args: Any, **kwargs: Any
) -> SimulationState:
    output = map_children(state, op, *args, **kwargs)
    return replace(state, members={**state.members, **output})


def __requested(
    state: SimulationState, datasets: str | Iterable[str] | None
) -> set[str]:
    if datasets is None:
        return set(state.members)
    requested = {datasets} if isinstance(datasets, str) else set(datasets)
    if not requested.issubset(state.members):
        raise ValueError(f"Unknown datasets {requested.difference(state.members)}")
    return requested


def __multi_dataset_members(
    state: SimulationState,
) -> dict[str, DatasetState | LightconeState] | None:
    """The members for automatic column matching, or None if any is a collection."""
    members: dict[str, DatasetState | LightconeState] = {}
    for name, child in state.members.items():
        if not isinstance(child, (DatasetState, LightconeState)):
            return None
        members[name] = child
    return members
