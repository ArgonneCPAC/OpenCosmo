from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import rustworkx as rx

from opencosmo.column.column import RawColumn
from opencosmo.dataset.graph import build_dependency_graph, contract_derived_columns


def get_all_required_columns(column_names: set[str], dependency_graph: rx.PyDiGraph):
    required_columns = set()
    node_map = {name: i for i, name in enumerate(dependency_graph.nodes())}
    for name in column_names:
        required_columns.add(name)
        ancestors = rx.ancestors(dependency_graph, node_map[name])
        required_columns.update(dependency_graph[i] for i in ancestors)

    return required_columns


if TYPE_CHECKING:
    from opencosmo.column.column import ConstructedColumn
    from opencosmo.handler.protocols import DataCache, DataHandler
    from opencosmo.index import DataIndex
    from opencosmo.units.handler import UnitHandler


def instantiate_dataset(
    column_producers: list[ConstructedColumn],
    column_names: set[str],
    raw_data_handler: DataHandler,
    cache: DataCache,
    unit_handler: UnitHandler,
    unit_kwargs: dict[str, Any],
    metadata_columns: list[str] | None = None,
    sort_by: tuple[str, bool] | None = None,
):
    column_names = copy(column_names)
    if sort_by is not None:
        column_names.add(sort_by[0])

    dependency_graph = build_dependency_graph(column_producers)
    all_required_columns = get_all_required_columns(column_names, dependency_graph)
    cached_data = cache.get_data(all_required_columns)
    converted_cached_data = unit_handler.apply_unit_conversions(
        cached_data, unit_kwargs
    )

    if converted_cached_data:
        cache.add_data(converted_cached_data, {}, push_up=False)
        cached_data |= converted_cached_data

    raw_columns = [
        col
        for col in column_producers
        if isinstance(col, RawColumn)
        and col.name not in cached_data
        and col.name in all_required_columns
    ]

    raw_data = raw_data_handler.get_data(set(col.name for col in raw_columns))
    for column in raw_columns:
        if column.alias is None:
            continue
        raw_data[column.alias] = raw_data[column.name]

    raw_data = unit_handler.apply_raw_units(raw_data, unit_kwargs)
    new_derived_columns = build_derived_columns(
        column_producers,
        all_required_columns,
        cached_data | raw_data,
        dependency_graph,
        raw_data_handler.index,
    )
    if raw_data:
        cache.add_data(raw_data, {}, push_up=True)
    converted_raw_data = unit_handler.apply_unit_conversions(raw_data, unit_kwargs)
    if converted_raw_data:
        cache.add_data(converted_raw_data, {}, push_up=False)
        raw_data |= converted_raw_data

    data = cached_data | raw_data | new_derived_columns
    data |= get_metadata_columns(raw_data_handler, cache, metadata_columns)
    return sort_data(data, sort_by)


def build_derived_columns(
    column_producers: list[ConstructedColumn],
    column_names: set[str],
    data: dict[str, np.ndarray],
    dependency_graph: rx.PyDiGraph,
    index: DataIndex,
):
    dependency_graph = contract_derived_columns(
        dependency_graph, column_names, column_producers
    )
    new_derived: dict[str, ConstructedColumn] = {}
    for colidx in rx.topological_sort(dependency_graph):
        column = dependency_graph[colidx]
        if isinstance(column, str):
            assert column in data or column not in column_names
            continue
        produces = column.produces
        if all(name in data for name in produces):
            continue
        output = column.evaluate(
            data | new_derived, index[1] if isinstance(index, tuple) else None
        )
        if isinstance(output, dict):
            new_derived |= output
        else:
            new_derived[column.name] = output
    return new_derived


def get_metadata_columns(
    raw_data_handler: DataHandler, cache: DataCache, metadata_columns: list[str] | None
):
    if metadata_columns is None:
        return {}
    metadata = cache.get_data(metadata_columns)
    additional_metadata_columns_to_fetch = set(metadata_columns).difference(
        metadata.keys()
    )
    metadata |= (
        raw_data_handler.get_metadata(additional_metadata_columns_to_fetch) or {}
    )

    return metadata


def sort_data(data: dict[str, np.ndarray], sort_by: tuple[str, bool] | None):
    if sort_by is None:
        return data
    sort_column = data[sort_by[0]]
    order = np.argsort(sort_column)
    if sort_by[1]:
        order = order[::-1]

    return {key: value[order] for key, value in data.items()}
