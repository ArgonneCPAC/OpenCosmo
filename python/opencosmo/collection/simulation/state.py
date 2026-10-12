from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, TypeAlias

from opencosmo.collection.simulation import matching
from opencosmo.dataset.state import DatasetState
from opencosmo.utils import normalize_kwarg_name

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opencosmo.collection.lightcone.state import LightconeState
    from opencosmo.collection.structure.state import StructureState
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.mapping.mapping import DatasetMatchSet

    SimulationChild: TypeAlias = DatasetState | LightconeState | StructureState


@dataclass(frozen=True)
class SimulationState:
    """
    Backend state for a SimulationCollection. When ``aligned`` is False, row
    operations have been applied to the active match source but not yet to the
    other members; they are realigned when ``children`` is first read.
    Construct with ``from_parts``.
    """

    members: Mapping[str, SimulationChild]
    match_set: DatasetMatchSet | None = None
    match_source: str | None = None
    aligned: bool = True

    def __post_init__(self):
        if self.match_source is not None and self.match_set is None:
            raise RuntimeError("A match source requires a match set")
        if self.match_source is not None and self.match_source not in self.members:
            raise RuntimeError(f"Match source {self.match_source} is not a member")
        if not self.aligned and self.match_source is None:
            raise RuntimeError("Only a matched collection can have pending members")

    @cached_property
    def children(self) -> Mapping[str, SimulationChild]:
        # cached_property writes to the instance __dict__, bypassing the frozen
        # __setattr__. The value is a pure function of the fields, and
        # dataclasses.replace does not carry it over.
        if self.aligned:
            return self.members
        assert self.match_set is not None and self.match_source is not None
        return matching.align(self.match_set, self.match_source, self.members)

    def __len__(self) -> int:
        return len(self.members)

    @property
    def header(self) -> dict[str, OpenCosmoHeader]:
        return {name: child.header for name, child in self.members.items()}

    @property
    def dtype(self) -> dict[str, str]:
        return {
            name: str(child.header.file.data_type)
            for name, child in self.members.items()
        }


def from_parts(
    members: Mapping[str, SimulationChild],
    match_set: DatasetMatchSet | None = None,
) -> SimulationState:
    """
    Build an unmatched simulation state. Member names are normalized, and a
    match set may only be paired with dataset members.
    """
    if match_set is not None and not all(
        isinstance(member, DatasetState) for member in members.values()
    ):
        raise ValueError(
            "Dataset matching is only supported for simple datasets (no collections)"
        )
    return SimulationState(
        {normalize_kwarg_name(name): member for name, member in members.items()},
        match_set,
    )


def public_keys(state: SimulationState) -> list[str]:
    """The keys a user sees: member names, in order."""
    return list(state.members)


def view(state: SimulationState, key: str) -> SimulationChild:
    """The value a user sees for ``key``, realigned if a match is pending."""
    children = state.children
    if key not in children:
        raise KeyError(f"Dataset {key} not found in collection.")
    return children[key]
