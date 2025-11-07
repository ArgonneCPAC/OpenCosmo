from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import networkx as nx

if TYPE_CHECKING:
    import numpy as np

    from opencosmo.column.cache import ColumnCache
    from opencosmo.column.column import DerivedColumn
    from opencosmo.dataset.handler import Hdf5Handler
    from opencosmo.units.handler import UnitHandler


def validate_derived_columns(
    derived_columns: dict[str, DerivedColumn],
    known_raw_columns: set[str],
    unit_handler: UnitHandler,
):
    dependency_graph = nx.DiGraph(
        {colname: dc.requires() for colname, dc in derived_columns.items()}
    )
    sources = set(
        map(
            lambda node: node[0],
            filter(
                lambda node: node[1] == 0 and node[0] in derived_columns,
                dependency_graph.out_degree,
            ),
        )
    )
    if missing := sources.difference(known_raw_columns.union(derived_columns)):
        raise ValueError(f"Columns {missing} do not exist in this dataset!")

    if cycles := list(nx.simple_cycles(dependency_graph)):
        raise ValueError(
            f"Found a cycle of derived columns which depend on each other! Cycle: {cycles[0]}"
        )

    dependency_graph.remove_nodes_from(known_raw_columns)
    units = unit_handler.base_units
    for derived_column_name in nx.topological_sort(dependency_graph):
        units[derived_column_name] = derived_columns[derived_column_name].get_units(
            units
        )

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

    dependencies = {name: derived_columns[name].requires() for name in column_names}
    dependency_graph = nx.DiGraph(dependencies)
    for _ in range(10):
        additional_derived = set(
            map(
                lambda node: node[0],
                filter(
                    lambda node: node[1] == 0 and node[0] in derived_columns,
                    dependency_graph.out_degree,
                ),
            )
        )
        if not additional_derived:
            break

        additional_dependencies = {
            colname: derived_columns[colname].requires()
            for colname in additional_derived
        }
        dependency_graph.update(edges=nx.DiGraph(additional_dependencies))

    dependency_graph = dependency_graph.reverse()
    cached_data = cache.get_columns(dependency_graph.nodes)
    updated_cached_data = unit_handler.apply_unit_conversions(cached_data, unit_kwargs)
    columns_to_fetch = (
        set(dependency_graph.nodes)
        .intersection(hdf5_handler.columns)
        .difference(cached_data.keys())
    )

    raw_data = hdf5_handler.get_data(columns_to_fetch)
    data = cached_data | unit_handler.apply_units(raw_data, unit_kwargs)
    for colname in nx.topological_sort(dependency_graph):
        if colname in data:
            continue
        data[colname] = derived_columns[colname].evaluate(data)

    return data
