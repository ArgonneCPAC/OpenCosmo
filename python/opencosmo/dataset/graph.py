from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

import rustworkx as rx

from opencosmo.column.column import RawColumn

if TYPE_CHECKING:
    from uuid import UUID

    import astropy.units as u

    from opencosmo.column.column import ConstructedColumn
    from opencosmo.units.handler import UnitHandler


def validate_column_producers(
    producers: list[ConstructedColumn], unit_handler: UnitHandler
):
    """
    Validate the network of column producers.
    """
    dependency_graph = build_dependency_graph(producers)

    if cycle := rx.digraph_find_cycle(dependency_graph):
        all_nodes: set[int] = reduce(
            lambda known, edge: known.union(edge), cycle, set()
        )
        names = [dependency_graph[i].produces for i in all_nodes]
        raise ValueError(f"Found columns that depend on each other! Columns: {names}")

    for i in range(dependency_graph.num_nodes()):
        if dependency_graph.in_degree(i):
            continue
        node = dependency_graph[i]
        if not isinstance(node, RawColumn):
            raise ValueError(
                f"Tried to derive columns from unknown columns: {node.produces}"
            )

    return get_derived_units(dependency_graph, unit_handler.base_units)


def build_dependency_graph(
    producers: list[ConstructedColumn],
) -> rx.PyDiGraph:
    graph = rx.PyDiGraph()
    uuid_to_node: dict[UUID, int] = {}

    for producer in producers:
        node_idx = graph.add_node(producer)
        uuid_to_node[producer.uuid] = node_idx

    for producer in producers:
        produces_idx = uuid_to_node[producer.uuid]
        if not producer.requires.issubset(uuid_to_node.keys()):
            raise ValueError(
                f"Producer {producer.produces} depends on an unknown producer UUID."
            )
        new_edges = (
            (uuid_to_node[dep_uuid], produces_idx) for dep_uuid in producer.requires
        )
        graph.add_edges_from_no_data(new_edges)

    return graph


def get_derived_units(
    dependency_graph: rx.PyDiGraph,
    units: dict[str, u.Unit],
):
    new_units: dict[str, u.Unit | None] = {}
    for node_idx in rx.topological_sort(dependency_graph):
        node = dependency_graph[node_idx]
        if isinstance(node, RawColumn):
            continue
        column_units = node.get_units(units | new_units)
        if not isinstance(column_units, dict):
            column_units = {prod: column_units for prod in node.produces}
        new_units |= column_units
    return new_units
