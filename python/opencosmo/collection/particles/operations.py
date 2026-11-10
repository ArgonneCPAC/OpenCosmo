"""
Standalone transformations of a ParticleState. Each one applies the matching
dataset operation to the particle types it concerns and returns a new state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from opencosmo.collection.particles.state import ParticleState
from opencosmo.column.column import DerivedScalarValue
from opencosmo.column.select import do_multi_dataset_drops, do_multi_dataset_selections
from opencosmo.dataset import operations as dsops
from opencosmo.io.schema import FileEntry
from opencosmo.io.schema import make_schema as make_collection_schema

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    import astropy.units as u

    from opencosmo.dataset.state import DatasetState
    from opencosmo.io.iopen import DatasetTarget
    from opencosmo.io.schema import Schema
    from opencosmo.spatial.protocols import Region


def open_state(
    targets: list[DatasetTarget],
    *,
    index_kind: str,
    is_empty_ref: bool,
    open_kwargs: dict[str, Any],
) -> ParticleState:
    """Open every particle type of a snapshot particle file as its own dataset.

    Under MPI, the partition hook in ``plugins.py`` gives every type the same
    region on each rank.
    """
    from opencosmo.io.index_spec import index_spec_for
    from opencosmo.io.iopen import open_dataset

    index = index_spec_for(index_kind, is_empty_ref, is_source=True)
    members = {
        target.name: open_dataset(target, index, open_kwargs=open_kwargs)
        for target in sorted(targets, key=lambda target: target.name)
    }
    return ParticleState(members)


def bound(
    state: ParticleState, region: Region, select_by: str | None = None
) -> ParticleState:
    return __map(
        state, lambda member: dsops.bound(member, region, select_by, snap=True)
    )


def with_units(
    state: ParticleState,
    convention: str | None,
    conversions: dict[u.UnitBase, u.UnitBase],
    **columns: u.UnitBase,
) -> ParticleState:
    """Apply a unit convention and conversions to every particle type.

    A column conversion applies to each type that has the column. A column that
    no type has raises ``ValueError``.
    """
    known = {name for member in state.members.values() for name in member.columns}
    missing = set(columns) - known
    if missing:
        raise ValueError(
            f"No particle type in this collection has the columns {sorted(missing)}"
        )

    def convert(member: DatasetState) -> DatasetState:
        names = set(member.columns)
        own = {name: unit for name, unit in columns.items() if name in names}
        return dsops.with_units(member, convention, conversions, **own)

    return __map(state, convert)


def select(
    state: ParticleState,
    select_args: tuple[str | Iterable[str], ...],
    select_kwargs: Mapping[str, Any],
    mode: Literal["local", "global"] = "global",
) -> ParticleState:
    """Select columns, matched to the particle types that have them or, when a
    keyword names a particle type, from that type explicitly."""
    if not select_args and not select_kwargs:
        return state

    if select_args or not set(select_kwargs).intersection(state.members):
        if any(isinstance(c, DerivedScalarValue) for c in select_kwargs.values()):
            raise ValueError(
                "Scalar values cannot be retrieved from a ParticleCollection "
                "directly. Use `collection[particle_type].select(scalar_expression)`"
            )
        return ParticleState(
            do_multi_dataset_selections(state.members, select_args, select_kwargs, mode)
        )

    members = dict(state.members)
    for name, columns in select_kwargs.items():
        member = __member(state, name)
        if isinstance(columns, dict):
            args = columns.get("columns", [])
            derived = columns.get("derived_columns", {})
        else:
            args = columns
            derived = {}
        if any(isinstance(c, DerivedScalarValue) for c in derived.values()):
            raise ValueError(
                "Scalar values cannot be retrieved from a ParticleCollection "
                f"directly. Use `collection[{name!r}].select(scalar_expression)`"
            )
        members[name] = dsops.select(member, args, mode=mode, **derived)
    return ParticleState(members)


def drop(
    state: ParticleState,
    drop_args: tuple[str | Iterable[str], ...],
    columns_to_drop: Mapping[str, str | Iterable[str]],
) -> ParticleState:
    """Drop columns from the particle types that have them, and from explicitly
    named particle types."""
    if drop_args:
        state = ParticleState(do_multi_dataset_drops(state.members, drop_args))
    if not columns_to_drop:
        return state
    members = dict(state.members)
    for name, columns in columns_to_drop.items():
        members[name] = dsops.drop(__member(state, name), columns)
    return ParticleState(members)


def with_datasets(state: ParticleState, datasets: str | Iterable[str]) -> ParticleState:
    requested = {datasets} if isinstance(datasets, str) else set(datasets)
    unknown = requested - set(state.members)
    if unknown:
        raise ValueError(
            f"Unknown particle types {sorted(unknown)}. "
            f"Available types: {sorted(state.members)}"
        )
    if not requested:
        raise ValueError("A particle collection needs at least one particle type")
    return ParticleState(
        {name: member for name, member in state.members.items() if name in requested}
    )


def evaluate_on_dataset(
    state: ParticleState,
    func: Callable,
    dataset: str,
    vectorize: bool,
    format: str,
    insert: bool,
    batch_size: int,
    allow_overwrite: bool,
    **evaluate_kwargs: Any,
) -> ParticleState | dict[str, Any] | None:
    result = dsops.evaluate(
        __member(state, dataset),
        func,
        vectorize=vectorize,
        insert=insert,
        format=format,
        batch_size=batch_size,
        allow_overwrite=allow_overwrite,
        **evaluate_kwargs,
    )
    if not insert:
        return result
    return ParticleState({**state.members, dataset: result})


def make_schema(state: ParticleState, path: str) -> Schema:
    """Schema for writing the collection: one dataset group per particle type,
    each with its own header and spatial index."""
    children = {
        name: dsops.make_schema(member, f"{path}/{name}")
        for name, member in state.members.items()
    }
    return make_collection_schema(
        path.split("/")[-1], FileEntry.PARTICLE_COLLECTION, children=children
    )


def __member(state: ParticleState, name: str) -> DatasetState:
    if name not in state.members:
        raise ValueError(
            f"Particle type {name} not found in collection. "
            f"Available types: {sorted(state.members)}"
        )
    return state.members[name]


def __map(
    state: ParticleState, operation: Callable[[DatasetState], DatasetState]
) -> ParticleState:
    return ParticleState(
        {name: operation(member) for name, member in state.members.items()}
    )
