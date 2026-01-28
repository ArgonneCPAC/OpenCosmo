from __future__ import annotations

from functools import reduce
from itertools import product
from typing import TYPE_CHECKING, Mapping, Optional

import rustworkx as rx

if TYPE_CHECKING:
    import astropy.units as u
    import numpy as np

    from opencosmo.column.cache import ColumnCache
    from opencosmo.column.column import ConstructedColumn
    from opencosmo.dataset.handler import Hdf5Handler
    from opencosmo.index import DataIndex
    from opencosmo.units.handler import UnitHandler


def build_dependency_graph(
    derived_columns: Mapping[str, ConstructedColumn],
    names_to_keep: Optional[set[str]] = None,
):
    dependency_graph = rx.PyDiGraph()
    all_requires: set[str] = reduce(
        lambda known, dc: known.union(dc.requires), derived_columns.values(), set()
    )
    nodeidx = dependency_graph.add_nodes_from(all_requires)
    nodemap = {name: idx for (name, idx) in zip(all_requires, nodeidx)}

    for target, derived_column in derived_columns.items():
        requires = derived_column.requires
        produces = derived_column.produces
        if produces is None:
            produces = set((target,))
        to_add = list(filter(lambda p: p not in nodemap, produces))
        new_map = dependency_graph.add_nodes_from(to_add)
        nodemap.update({name: idx for (name, idx) in zip(to_add, new_map)})

        requires_idx = tuple(nodemap[r] for r in requires)
        produces_idx = tuple(nodemap[r] for r in produces)

        dependency_graph.add_edges_from_no_data(product(requires_idx, produces_idx))

    if names_to_keep is not None:
        nodes_to_keep = reduce(
            lambda acc, name: acc.union(rx.ancestors(dependency_graph, nodemap[name])),
            names_to_keep,
            {nodemap[name] for name in names_to_keep},
        )
        names_to_keep = {dependency_graph[n] for n in nodes_to_keep}
        dependency_graph = dependency_graph.subgraph(list(nodes_to_keep))
        derived_columns = {
            name: dc for name, dc in derived_columns.items() if name in names_to_keep
        }

    return dependency_graph


def replace_multi_producers(
    graph: rx.PyDiGraph, derived_columns: Mapping[str, ConstructedColumn]
):
    """
    Some derived columns actually produce multiple outputs. At this stage, the dependency
    graph is working solely with actual column names, meaning if any of those columns is
    produced by one of these "multi-produces" they will not be in the derived_columns
    dictionary and therefore cannot be instantiated. This function replaces such
    columns with the name of the derived_column that produces them.
    """

    node_map = {name: i for i, name in enumerate(graph.nodes())}
    missing = set(derived_columns.keys()).difference(node_map.keys())
    if not missing:
        return graph
    for missing_column in missing:
        missing_column_produces = derived_columns[missing_column].produces
        if missing_column_produces is None:
            continue
        outputs = [
            node_map[name] for name in missing_column_produces if name in node_map
        ]
        graph.contract_nodes(outputs, missing_column)
    return graph


def validate_derived_columns(
    derived_columns: dict[str, ConstructedColumn],
    known_raw_columns: set[str],
    units: dict[str, u.Unit],
):
    """
    Validate the network of derived columns. This
    """
    dependency_graph = build_dependency_graph(derived_columns)
    if cycle := rx.digraph_find_cycle(dependency_graph):
        all_nodes: set[int] = reduce(
            lambda known, edge: known.union(edge), cycle, set()
        )
        names = [dependency_graph[i] for i in all_nodes]
        raise ValueError(
            f"Found derived columns that depend on each other! Columns: {names}"
        )

    sources = set(
        filter(
            lambda i: not dependency_graph.in_degree(i),
            range(dependency_graph.num_nodes()),
        )
    )
    source_names = map(lambda i: dependency_graph[i], sources)
    if missing := set(source_names).difference(known_raw_columns):
        raise ValueError(f"Tried to derive columns from unknown columns: {missing}")

    dependency_graph = replace_multi_producers(dependency_graph, derived_columns)
    validate_dependency_graph(
        dependency_graph, known_raw_columns, set(derived_columns.keys())
    )

    return validate_derived_units(dependency_graph, derived_columns, units)


def validate_dependency_graph(
    dependency_graph: rx.PyDiGraph,
    known_raw_columns: set[str],
    derived_columns: set[str],
):
    expected = set(dependency_graph.nodes())
    known = known_raw_columns.union(derived_columns)
    missing = expected.difference(known)
    assert len(missing) == 0


def validate_derived_units(
    dependency_graph: rx.PyDiGraph,
    derived_columns: dict[str, ConstructedColumn],
    units: dict[str, u.Unit],
):
    output_units: dict[str, Optional[u.Unit]] = {}
    for node in rx.topological_sort(dependency_graph):
        node_name = dependency_graph[node]
        if node_name in units:
            continue
        new_units = derived_columns[node_name].get_units(units)
        if not isinstance(new_units, dict):
            new_units = {node_name: new_units}
        units |= new_units
        output_units |= new_units
    return output_units


def build_derived_columns(
    all_derived_columns: dict[str, ConstructedColumn],
    derived_columns_to_get: set[str],
    cache: ColumnCache,
    hdf5_handler: Hdf5Handler,
    unit_handler: UnitHandler,
    unit_kwargs: dict,
    index: DataIndex,
) -> dict[str, np.ndarray]:
    """
    Build any derived columns that are present in this dataset. Also returns any columns that
    had to be instantiated in order to build these derived columns.
    """
    if not derived_columns_to_get:
        return {}

    column_names: set[str] = reduce(
        lambda known, dc: known.union(dc[1].produces)
        if dc[1].produces is not None
        else known.union((dc[0],)),
        all_derived_columns.items(),
        set(),
    )

    dependency_graph = build_dependency_graph(
        all_derived_columns, derived_columns_to_get
    )
    cached_data = cache.get_columns(dependency_graph.nodes())
    cached_data |= unit_handler.apply_unit_conversions(cached_data, unit_kwargs)

    additional_derived = column_names.difference(cached_data.keys())

    if not additional_derived:
        return cached_data

    columns_to_fetch = (
        set(dependency_graph.nodes())
        .intersection(hdf5_handler.columns)
        .difference(cached_data.keys())
    )

    raw_data = hdf5_handler.get_data(columns_to_fetch)
    data = cached_data | unit_handler.apply_units(raw_data, unit_kwargs)

    dependency_graph = replace_multi_producers(dependency_graph, all_derived_columns)
    new_derived: dict[str, np.ndarray] = {}

    for colidx in rx.topological_sort(dependency_graph):
        colname = dependency_graph[colidx]
        if colname in data:
            continue
        derived_column = all_derived_columns[colname]
        produces = derived_column.produces
        if produces is None:
            produces = set((colname,))
        if all(name in data for name in produces):
            continue
        output = derived_column.evaluate(data, index)
        if isinstance(output, dict):
            data |= output
            new_derived |= output
        else:
            data[colname] = output
            new_derived[colname] = output

    if new_derived:
        cache.add_data(new_derived)

    return data | new_derived
