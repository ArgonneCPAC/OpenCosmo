"""
Standalone transformations of a ParticleState. Each one applies the matching
dataset operation to every particle type and returns a new state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opencosmo.collection.particles.state import ParticleState
from opencosmo.dataset import operations as dsops

if TYPE_CHECKING:
    from collections.abc import Callable

    import astropy.units as u

    from opencosmo.dataset.state import DatasetState
    from opencosmo.io.iopen import DatasetTarget
    from opencosmo.spatial.protocols import Region


def open_state(
    targets: list[DatasetTarget],
    *,
    index_kind: str,
    is_empty_ref: bool,
    open_kwargs: dict[str, Any],
) -> ParticleState:
    """Open every particle type of a snapshot particle file as its own dataset.

    Each type is indexed by its own octree, so under MPI each one is partitioned
    on its own, as a standalone dataset would be.
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
    return __map(state, lambda member: dsops.bound(member, region, select_by))


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


def __map(
    state: ParticleState, operation: Callable[[DatasetState], DatasetState]
) -> ParticleState:
    return ParticleState(
        {name: operation(member) for name, member in state.members.items()}
    )
