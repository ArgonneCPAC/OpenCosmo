from __future__ import annotations

from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING, Iterable

import rustworkx as rx

if TYPE_CHECKING:
    import numpy as np

    from opencosmo.column.cache import ColumnCache
    from opencosmo.column.column import DerivedColumn
    from opencosmo.dataset.handler import Hdf5Handler
    from opencosmo.units.handler import UnitHandler


def build_dependency_graph(derived_columns: dict[str, DerivedColumn]):
    dependency_graph = rx.PyDiGraph()
    all_requires: set[str] = reduce(
        lambda known, dc: known.union(dc.requires()), derived_columns.values(), set()
    )
    all_columns = {
        colname: i
        for i, colname in enumerate(all_requires.union(derived_columns.keys()))
    }
    _ = dependency_graph.add_nodes_from(all_columns.keys())

    for target, derived_column in derived_columns.items():
        dependency_graph.add_edges_from_no_data(
            (all_columns[source], all_columns[target])
            for source in derived_column.requires()
        )

    return dependency_graph, set(
        all_columns[target] for target in derived_columns.keys()
    )


def validate_derived_columns(
    derived_columns: dict[str, DerivedColumn],
    known_raw_columns: set[str],
    unit_handler: UnitHandler,
):
    """
    Validate the network of derived columns. This
    """
    dependency_graph, targets = build_dependency_graph(derived_columns)
    if cycle := rx.digraph_find_cycle(dependency_graph):
        names = [dependency_graph[i] for i in cycle]
        raise ValueError(
            f"Found derived columns that depend on each other! Columns: {names}"
        )

    nodes_to_keep = reduce(
        lambda anc, target: anc.union(rx.ancestors(dependency_graph, target)),
        targets,
        targets,
    )
    dependency_graph = dependency_graph.subgraph(list(nodes_to_keep))
    sources = set(
        filter(
            lambda i: not dependency_graph.in_degree(i),
            range(dependency_graph.num_nodes()),
        )
    )
    source_names = map(lambda i: dependency_graph[i], sources)
    if missing := set(source_names).difference(known_raw_columns):
        raise ValueError(f"Tried to derive columns from unknown columns: {missing}")

    units = {
        dependency_graph[source]: unit_handler.base_units[dependency_graph[source]]
        for source in sources
    }

    for column_index in rx.topological_sort(dependency_graph):
        if column_index in sources:
            continue
        units[dependency_graph[column_index]] = derived_columns[
            dependency_graph[column_index]
        ].get_units(units)

    new_unit_handler = unit_handler.with_new_columns(
        **{
            derived_column_name: units[derived_column_name]
            for derived_column_name in derived_columns
        }
    )
    return new_unit_handler


def build_derived_columns(
    column_names: set[str],
    derived_columns: dict[str, DerivedColumn],
    cache: ColumnCache,
    hdf5_handler: Hdf5Handler,
    unit_handler: UnitHandler,
    unit_kwargs: dict,
) -> dict[str, np.ndarray]:
    """
    Build any derived columns that are present in this dataset. Also returns any columns that
    had to be instantiated in order to build these derived columns.
    """
    if not derived_columns:
        return {}
    cached_data = cache.get_columns(column_names)
    additional_derived = column_names.difference(cached_data.keys())

    if not additional_derived:
        return cached_data

    dependency_graph, targets = build_dependency_graph(derived_columns)
    cached_data |= cache.get_columns(dependency_graph.nodes())
    cached_data = unit_handler.apply_unit_conversions(cached_data, unit_kwargs)

    columns_to_fetch = (
        set(dependency_graph.nodes())
        .intersection(hdf5_handler.columns)
        .difference(cached_data.keys())
    )

    raw_data = hdf5_handler.get_data(columns_to_fetch)
    data = cached_data | unit_handler.apply_units(raw_data, unit_kwargs)

    for colidx in rx.topological_sort(dependency_graph):
        colname = dependency_graph[colidx]
        if colname in data:
            continue
        data[colname] = derived_columns[colname].evaluate(data)

    return data
