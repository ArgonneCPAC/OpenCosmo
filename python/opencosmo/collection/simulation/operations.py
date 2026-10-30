"""
Simulation collection operations on SimulationState.

Row operations on a matched collection apply only to the active source and
defer realigning the other members (see ``SimulationState.children``). All
other operations map over the members and keep the current alignment, since
matching only reads raw row indices.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from opencosmo.collection.lightcone import operations as lcops
from opencosmo.collection.lightcone.state import LightconeState
from opencosmo.collection.simulation import matching
from opencosmo.collection.structure import operations as scops
from opencosmo.collection.structure.state import StructureState
from opencosmo.column.select import do_multi_dataset_drops, do_multi_dataset_selections
from opencosmo.dataset import operations as dsops
from opencosmo.dataset.state import DatasetState
from opencosmo.io.schema import FileEntry
from opencosmo.io.schema import make_schema as make_collection_schema

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import astropy.units as u
    import numpy as np

    from opencosmo.collection.simulation.state import SimulationChild, SimulationState
    from opencosmo.column.column import ColumnMask, ConstructedColumn
    from opencosmo.index import DataIndex
    from opencosmo.io.schema import Schema
    from opencosmo.spatial.protocols import Region


def __child_op(child: SimulationChild, name: str) -> Callable[..., Any]:
    """
    The operation ``name`` for ``child``. The dataset, lightcone, and structure
    operation modules share operation names and signatures.
    """
    if isinstance(child, DatasetState):
        return getattr(dsops, name)
    if isinstance(child, LightconeState):
        return getattr(lcops, name)
    return getattr(scops, name)


def __map_children(
    state: SimulationState,
    op: str,
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
        output[name] = __child_op(child, op)(
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
    return __row_operation(state, "filter", datasets, *masks, **kwargs)


def bound(
    state: SimulationState, region: Region, select_by: str | None = None
) -> SimulationState:
    return __row_operation(state, "bound", None, region, select_by)


def sort_by(state: SimulationState, column: str, invert: bool) -> SimulationState:
    return __row_operation(state, "sort_by", None, column, invert)


def take(state: SimulationState, n: int, at: str, mode: str) -> SimulationState:
    return __row_operation(state, "take", None, n, at, mode)


def take_range(
    state: SimulationState, start: int, end: int, mode: str
) -> SimulationState:
    return __row_operation(state, "take_range", None, start, end, mode)


def take_rows(state: SimulationState, rows: DataIndex) -> SimulationState:
    return __row_operation(state, "take_rows", None, rows)


# Column operations


def select(
    state: SimulationState, *args: str | Iterable[str], **kwargs: Any
) -> SimulationState:
    datasets = __multi_dataset_members(state)
    if datasets is None:
        return __map(state, "select", *args, **kwargs)
    return replace(state, members=do_multi_dataset_selections(datasets, args, kwargs))


def drop(
    state: SimulationState, *args: str | Iterable[str], **kwargs: Any
) -> SimulationState:
    datasets = __multi_dataset_members(state)
    if datasets is None:
        return __map(state, "drop", *args, **kwargs)

    output = do_multi_dataset_drops(datasets, args)
    for name, columns in kwargs.items():
        if name not in output:
            raise ValueError(f"Dataset {name} not found in collection.")
        output[name] = __child_op(output[name], "drop")(output[name], columns)
    return replace(state, members=output)


def with_new_columns(
    state: SimulationState,
    *args: Any,
    datasets: str | Iterable[str] | None = None,
    descriptions: str | dict[str, str] | None = None,
    allow_overwrite: bool = False,
    **new_columns: ConstructedColumn | np.ndarray,
) -> SimulationState:
    # New columns may be arrays sized to the realigned rows.
    if descriptions is None:
        descriptions = {}
    return __map(
        __resolved(state),
        "with_new_columns",
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
        "with_units",
        convention,
        conversions=conversions,
        **columns,
    )


# Evaluation and data access


def evaluate(
    state: SimulationState,
    func: Callable,
    datasets: str | Iterable[str] | None,
    format: str,
    vectorize: bool,
    insert: bool,
    allow_overwrite: bool,
    **evaluate_kwargs: Any,
) -> SimulationState | dict[str, Any]:
    """
    Evaluate ``func`` on the requested members after realignment. With
    ``insert`` the results replace those members; otherwise they are returned
    keyed by member name.
    """
    state = __resolved(state)
    results = __map_children(
        state,
        "evaluate",
        func,
        datasets=datasets,
        vectorize=vectorize,
        insert=insert,
        format=format,
        allow_overwrite=allow_overwrite,
        **evaluate_kwargs,
    )
    if insert:
        return replace(state, members={**state.members, **results})
    return results


def get_data(
    state: SimulationState, format: str, unpack: bool, wrap_single: bool
) -> dict[str, Any]:
    if __multi_dataset_members(state) is None:
        raise TypeError(
            "get_data is only supported when this collection contains datasets!"
        )
    return {
        name: __child_op(child, "get_data")(child, format, unpack, wrap_single)
        for name, child in __resolved(state).members.items()
    }


# Matching and writing


def match(state: SimulationState, source: str) -> SimulationState:
    """
    Match every member to ``source``. Pending members are matched as they are,
    without realigning them to the previous source first.
    """
    if state.match_set is None:
        raise ValueError(
            "This SimulationCollection does not contain matching information!"
        )
    if source not in state.members:
        raise ValueError(
            f"This SimulationCollection does not have a simulation named {source}"
        )
    members = matching.align(state.match_set, source, state.members)
    return replace(state, members=members, match_source=source, aligned=True)


def clear_match(state: SimulationState) -> SimulationState:
    return replace(__resolved(state), match_source=None)


def make_schema(state: SimulationState, path: str) -> Schema:
    state = __resolved(state)
    children: dict[str, Schema] = {}
    new_uuids: dict[str, UUID] = {}
    indices: dict[str, DataIndex] = {}

    for name, child in state.members.items():
        children[name] = scops.child_schema(child, f"{path}/{name}", False)
        if isinstance(child, DatasetState):
            new_uuids[name] = UUID(
                cast("str", children[name].children["data"].attributes["main_uuid"])
            )
            indices[name] = child.raw_index
        elif isinstance(child, StructureState):
            new_uuids[name] = UUID(
                cast(
                    "str",
                    children[name]
                    .children[child.dtype]
                    .children["data"]
                    .attributes["main_uuid"],
                )
            )
            indices[name] = child.source.raw_index

    if state.match_set is not None and len(state) > 1:
        children["map"] = state.match_set.make_schema(
            new_uuids, indices, state.match_source
        )

    return make_collection_schema(
        "/", FileEntry.SIMULATION_COLLECTION, children=children
    )


def __row_operation(
    state: SimulationState,
    op: str,
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
            f"When working with a matched collection, {op} can only be called on the active source. Got datasets = {datasets}"
        )
    new_source = __child_op(state.members[source], op)(
        state.members[source], *args, **kwargs
    )
    return replace(state, members={**state.members, source: new_source}, aligned=False)


def __map(
    state: SimulationState, op: str, *args: Any, **kwargs: Any
) -> SimulationState:
    output = __map_children(state, op, *args, **kwargs)
    return replace(state, members={**state.members, **output})


def __resolved(state: SimulationState) -> SimulationState:
    return replace(state, members=state.children, aligned=True)


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
