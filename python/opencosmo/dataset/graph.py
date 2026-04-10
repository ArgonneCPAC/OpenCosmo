from __future__ import annotations

from functools import reduce
from itertools import product
from typing import TYPE_CHECKING

import rustworkx as rx

from opencosmo.column.column import RawColumn

if TYPE_CHECKING:
    import astropy.units as u

    from opencosmo.column.column import ConstructedColumn
    from opencosmo.units.handler import UnitHandler


def validate_column_producers(
    producers: list[ConstructedColumn], unit_handler: UnitHandler
):
    """
    Validate the network of column producers.
    """
    raw_columns = set(col.name for col in producers if isinstance(col, RawColumn))

    dependency_graph = build_dependency_graph(producers)
    if cycle := rx.digraph_find_cycle(dependency_graph):
        all_nodes: set[int] = reduce(
            lambda known, edge: known.union(edge), cycle, set()
        )
        names = [dependency_graph[i] for i in all_nodes]
        raise ValueError(f"Found columns that depend on each other! Columns: {names}")

    sources = set(
        filter(
            lambda i: not dependency_graph.in_degree(i),
            range(dependency_graph.num_nodes()),
        )
    )
    source_names = set(map(lambda i: dependency_graph[i], sources))
    if missing := set(source_names).difference(raw_columns):
        raise ValueError(f"Tried to derive columns from unknown columns: {missing}")

    return get_derived_units(dependency_graph, producers, unit_handler.base_units)


def build_dependency_graph(producers: list[ConstructedColumn]):
    dependency_graph = rx.PyDiGraph()
    all_requires: set[str] = reduce(
        lambda known, dc: known.union(dc.requires if dc.requires is not None else []),
        producers,
        set(),
    )
    nodeidx = dependency_graph.add_nodes_from(all_requires)
    nodemap = {name: idx for (name, idx) in zip(all_requires, nodeidx)}

    for column_producer in producers:
        requires = column_producer.requires
        produces = column_producer.produces
        assert produces is not None
        to_add = list(filter(lambda p: p not in nodemap, produces))
        new_map = dependency_graph.add_nodes_from(to_add)
        nodemap.update({name: idx for (name, idx) in zip(to_add, new_map)})
        if not requires:
            continue

        requires_idx = tuple(nodemap[r] for r in requires)
        produces_idx = tuple(nodemap[r] for r in produces)

        dependency_graph.add_edges_from_no_data(product(requires_idx, produces_idx))

    return dependency_graph


def contract_derived_columns(
    graph: rx.PyDiGraph,
    column_names: set[str],
    column_producers: list[ConstructedColumn],
):
    """
    Some derived columns actually produce multiple outputs. At this stage, the dependency
    graph is working solely with actual column names, meaning if any of those columns is
    produced by one of these "multi-produces" they will not be in the derived_columns
    dictionary and therefore cannot be instantiated. This function replaces such
    columns with the name of the derived_column that produces them.
    """
    node_map = {name: i for i, name in enumerate(graph.nodes())}
    nodes_to_keep: set[int] = set()
    for producer in column_producers:
        if producer.produces.intersection(column_names):
            produces_idx = {node_map[name] for name in producer.produces}
            nodes_to_keep = reduce(
                lambda acc, node: acc.union(rx.ancestors(graph, node)),
                produces_idx,
                nodes_to_keep,
            )
            nodes_to_keep.update(produces_idx)
    subgraph = graph.subgraph(list(nodes_to_keep))
    node_map = {name: i for i, name in enumerate(subgraph.nodes())}

    for producer in column_producers:
        if isinstance(producer, RawColumn):
            continue
        produces = producer.produces
        produces_index = [node_map[name] for name in produces if name in node_map]
        if produces_index:
            subgraph.contract_nodes(produces_index, producer)
    return subgraph


def get_derived_units(
    dependency_graph: rx.PyDiGraph,
    producers: list[ConstructedColumn],
    units: dict[str, u.Unit],
):
    dependency_graph = contract_derived_columns(
        dependency_graph, set(dependency_graph.nodes()), producers
    )

    new_units: dict[str, u.Unit | None] = {}
    for node in rx.topological_sort(dependency_graph):
        node = dependency_graph[node]
        if isinstance(node, str):
            continue
        column_units = node.get_units(units | new_units)
        if not isinstance(column_units, dict):
            column_units = {prod: column_units for prod in node.produces}
        new_units |= column_units
    return new_units
