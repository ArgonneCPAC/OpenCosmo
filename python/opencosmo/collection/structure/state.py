from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from opencosmo.collection.structure import links

if TYPE_CHECKING:
    from collections.abc import Mapping
    from uuid import UUID

    from opencosmo.collection.lightcone.state import LightconeState
    from opencosmo.dataset.state import DatasetState
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.mapping.mapping import DatasetMatchSet
    from opencosmo.spatial.protocols import Region

    Source = DatasetState | LightconeState


@dataclass(frozen=True)
class StructureState:
    """
    Backend state for a StructureCollection. ``linked`` is row-aligned with
    ``linked_source``, or with ``source`` when ``linked_source`` is None. Row
    operations on the source defer rebuilding the linked children until
    ``children`` is first read. Construct with ``from_parts``.
    """

    source: Source
    linked: Mapping[str, StructureChild]
    match_sets: Mapping[UUID, DatasetMatchSet]
    linked_source: Source | None = None
    hide_source: bool = False
    derived_columns: frozenset[str] = frozenset()

    @cached_property
    def children(self) -> Mapping[str, StructureChild]:
        # cached_property writes to the instance __dict__, bypassing the frozen
        # __setattr__. The value is a pure function of the fields, and
        # dataclasses.replace does not carry it over.
        if self.linked_source is None:
            return self.linked
        return links.rebuild(
            self.match_sets, self.linked_source, self.source, self.linked
        )

    def __len__(self) -> int:
        return len(self.source)

    @property
    def header(self) -> OpenCosmoHeader:
        return self.source.header

    @property
    def dtype(self) -> str:
        return str(self.source.header.dtype)

    @property
    def region(self) -> Region | None:
        return self.source.region

    @property
    def sorted_by(self) -> str | None:
        sort_key = self.source.sort_key
        return sort_key[0] if sort_key is not None else None

    @property
    def properties(self) -> list[str]:
        return self.source.columns


if TYPE_CHECKING:
    # Includes StructureCollection facades until the facade holds a StructureState.
    StructureChild = links.LinkTarget


def from_parts(
    source: Source,
    linked: Mapping[str, StructureChild],
    match_sets: Mapping[UUID, DatasetMatchSet],
    *,
    resolve: bool,
    hide_source: bool = False,
    derived_columns: frozenset[str] = frozenset(),
) -> StructureState:
    """
    Build a structure state. A nested galaxy collection is keyed as
    ``galaxies``. With ``resolve``, freshly opened children are aligned with
    ``source``; otherwise they must already be.
    """
    linked = dict(linked)
    if isinstance(linked.get("galaxy_properties"), StructureState):
        linked["galaxies"] = linked.pop("galaxy_properties")

    if linked:
        missing = [
            leaf.uuid
            for _, leaf in links.source_leaves(source)
            if leaf.uuid not in match_sets
        ]
        if missing:
            raise ValueError(
                "Structure collections require link match sets for every source "
                f"dataset, but none were found for {missing}"
            )

    if resolve:
        linked = links.resolve(match_sets, source, linked)

    return StructureState(
        source,
        linked,
        match_sets,
        hide_source=hide_source,
        derived_columns=derived_columns,
    )


def public_keys(state: StructureState) -> list[str]:
    """The keys a user sees: linked children first, then the source unless hidden."""
    keys = list(state.linked)
    if not state.hide_source:
        keys.append(state.dtype)
    return keys


def view(state: StructureState, key: str) -> StructureChild:
    """The value a user sees for ``key``. The source is reachable even when hidden."""
    if key == state.dtype:
        return state.source
    children = state.children
    if key not in children:
        raise KeyError(f"Dataset {key} not found in collection.")
    return children[key]
