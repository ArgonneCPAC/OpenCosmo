from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opencosmo._lib import graph as graphlib
from opencosmo.column.column import RawColumn

if TYPE_CHECKING:
    from collections.abc import Mapping
    from uuid import UUID

    import astropy.units as u

    from opencosmo._lib.graph import DependencyGraph
    from opencosmo.column.column import ConstructedColumn


def get_all_required_pairs(
    producers: Mapping[UUID, ConstructedColumn],
    columns_to_uuid: dict[str, UUID],
) -> set[tuple[UUID, str]]:
    """
    Return the full set of (producer_uuid, column_name) pairs needed to
    produce the requested columns, including all transitive dependencies.
    """
    required = build_dependency_graph(producers).order(set(columns_to_uuid.values()))
    pairs = {(uuid, name) for name, uuid in columns_to_uuid.items()}
    for uuid in required:
        pairs.update((uuid, name) for name in producers[uuid].produces)
    return pairs


def build_dependency_graph(
    producers: Mapping[UUID, ConstructedColumn],
) -> DependencyGraph:
    return graphlib.make_dependency_graph(
        {uuid: producer.requires for uuid, producer in producers.items()}
    )


def evaluate_producers(
    producers: Mapping[UUID, ConstructedColumn],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate a set of producers in topological order against a flat
    name-keyed input mapping, returning the produced columns by name.

    Each producer reads its dependencies by name from ``inputs`` (extended
    with what earlier producers in the graph have already produced). The
    flat-mapping form is intentional: callers that don't carry a full
    UUID-keyed cache (lightcone scope, ad-hoc evaluation against a
    materialized table) can just hand over the columns they have.

    Used by Lightcone.get_data to materialize scope-owned derived columns
    against the vstacked per-child data.
    """
    outputs: dict[str, Any] = {}
    for uuid in build_dependency_graph(producers).order():
        producer = producers[uuid]
        if isinstance(producer, RawColumn):
            continue
        # Scope producers evaluate over the stacked table, so there is no source index.
        result = producer.evaluate(inputs | outputs, None)  # type: ignore[arg-type]
        if not isinstance(result, dict):
            result = {next(iter(producer.produces)): result}
        outputs.update(result)
    return outputs


def get_derived_units(
    producers: Mapping[UUID, ConstructedColumn],
    units: dict[str, u.Unit],
) -> dict[str, u.Unit | None]:
    new_units: dict[str, u.Unit | None] = {}
    for uuid in build_dependency_graph(producers).order():
        producer = producers[uuid]
        if isinstance(producer, RawColumn):
            continue
        column_units = producer.get_units(units | new_units)
        if not isinstance(column_units, dict):
            column_units = {name: column_units for name in producer.produces}
        new_units |= column_units
    return new_units
