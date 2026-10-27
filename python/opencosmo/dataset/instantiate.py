from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opencosmo.column.column import EvaluatedColumn, RawColumn
from opencosmo.dataset.graph import get_all_required_pairs, get_dependency_graph

if TYPE_CHECKING:
    from collections.abc import Mapping
    from uuid import UUID

    import numpy as np

    from opencosmo._lib.graph import DependencyGraph
    from opencosmo.column.column import ConstructedColumn
    from opencosmo.handler.protocols import DataCache, DataHandler
    from opencosmo.index import DataIndex
    from opencosmo.units.handler import UnitHandler


def build_initial_uuid_data(
    column_producers: Mapping[UUID, ConstructedColumn],
    raw_data: dict[str, np.ndarray],
    cached_data: dict[UUID, dict[str, np.ndarray]],
) -> dict[UUID, dict[str, np.ndarray]]:
    """
    Merge cached and freshly-fetched raw data into UUID-keyed storage.
    Cached data is the starting point; raw data fills in any gaps.
    """
    uuid_data: dict[UUID, dict[str, np.ndarray]] = {**cached_data}
    for producer in column_producers.values():
        if not isinstance(producer, RawColumn) or producer.uuid in uuid_data:
            continue
        output_name = producer.alias or producer.name
        if output_name in raw_data:
            uuid_data[producer.uuid] = {output_name: raw_data[output_name]}
    return uuid_data


def build_derived_columns(
    columns_to_uuid: dict[str, UUID],
    uuid_data: dict[UUID, dict[str, np.ndarray]],
    dependency_graph: DependencyGraph,
    column_producers: Mapping[UUID, ConstructedColumn],
    index: DataIndex,
    cache: DataCache,
) -> dict[UUID, dict[str, np.ndarray]]:
    """
    Evaluate all derived producers needed to produce the requested columns,
    in topological order. Each producer's inputs are resolved by UUID via
    dep_map, so column-name shadowing cannot cause a derived column to
    receive data from the wrong producer.
    """
    new_derived: dict[UUID, dict[str, np.ndarray]] = {}
    to_cache: dict[UUID, dict[str, np.ndarray]] = {}
    for uuid in dependency_graph.order(set(columns_to_uuid.values())):
        producer = column_producers[uuid]
        if isinstance(producer, RawColumn) or uuid in uuid_data:
            continue

        all_data = uuid_data | new_derived
        input_data = {
            name: all_data[dep_uuid][name]
            for name, dep_uuid in (producer.dep_map or {}).items()
        }
        output: dict[str, np.ndarray]
        if isinstance(producer, EvaluatedColumn):
            output = producer.evaluate_for_storage(input_data, index)
        else:
            result = producer.evaluate(input_data, index)
            output = (
                result
                if isinstance(result, dict)
                else {next(iter(producer.produces)): result}
            )
        new_derived[uuid] = output
        if not producer.no_cache:
            to_cache[uuid] = output

    cache.add_data(to_cache, {})
    return new_derived


def __cache_raw_columns(
    raw_columns: list[RawColumn],
    raw_data: dict[str, Any],
    working_columns: dict[str, UUID],
    unit_handler: UnitHandler,
    unit_kwargs: dict[str, Any],
    cache: DataCache,
) -> dict[UUID, dict[str, Any]]:
    """
    Write freshly-fetched raw columns to the cache and return the merged
    (pre- and post-conversion) UUID-keyed data for merging into uuid_data.

    Pre-conversion data is pushed up to parent caches. Converted data is kept
    local (push_up=False) to avoid propagating dataset-specific unit conversions
    to parent caches, which could cause drift through repeated rounding.
    """
    raw_by_uuid: dict[UUID, dict[str, Any]] = {
        col.uuid: {col.alias or col.name: raw_data[col.alias or col.name]}
        for col in raw_columns
        if (col.alias or col.name) in raw_data
    }
    converted_by_uuid = unit_handler.apply_unit_conversions(raw_by_uuid, unit_kwargs)

    no_cache_uuids = {col.uuid for col in raw_columns if col.no_cache}
    cacheable = set(working_columns.values()) - no_cache_uuids
    cache.add_data(
        {uuid: data for uuid, data in raw_by_uuid.items() if uuid in cacheable},
        {},
        push_up=True,
    )
    cache.add_data(
        {uuid: data for uuid, data in converted_by_uuid.items() if uuid in cacheable},
        {},
        push_up=False,
    )

    return {
        uuid: (data | converted_by_uuid.get(uuid, {}))
        for uuid, data in raw_by_uuid.items()
    }


def instantiate_dataset(
    column_producers: Mapping[UUID, ConstructedColumn],
    columns_to_uuid: dict[str, UUID],
    raw_data_handler: DataHandler,
    cache: DataCache,
    unit_handler: UnitHandler,
    unit_kwargs: dict[str, Any],
    sort_by: str | None = None,
):
    # Extend working_columns with the sort column if it isn't already included.
    working_columns = dict(columns_to_uuid)
    if sort_by is not None and sort_by not in working_columns:
        for uuid, producer in column_producers.items():
            if sort_by in producer.produces:
                working_columns[sort_by] = uuid
                break

    required_pairs = get_all_required_pairs(column_producers, working_columns)
    dependency_graph = get_dependency_graph(column_producers)

    cached_data = cache.get_data(required_pairs)

    converted_cached = unit_handler.apply_unit_conversions(cached_data, unit_kwargs)
    if converted_cached:
        cache.add_data(converted_cached, {}, push_up=False)
        for uuid, col_data in converted_cached.items():
            cached_data.setdefault(uuid, {}).update(col_data)

    # Determine which raw columns still need to be fetched from the handler.
    cached_uuids = set(cached_data.keys())
    raw_columns = [
        col
        for col in column_producers.values()
        if isinstance(col, RawColumn)
        and col.uuid not in cached_uuids
        and col.name in {name for (_, name) in required_pairs}
    ]
    raw_data = raw_data_handler.get_data({col.name for col in raw_columns})
    for column in raw_columns:
        if column.alias is None:
            continue
        raw_data[column.alias] = raw_data[column.name]

    raw_data = unit_handler.apply_raw_units(raw_data, unit_kwargs)

    uuid_data = build_initial_uuid_data(column_producers, raw_data, cached_data)
    new_derived = build_derived_columns(
        working_columns,
        uuid_data,
        dependency_graph,
        column_producers,
        raw_data_handler.index,
        cache,
    )

    uuid_data |= new_derived

    uuid_data |= __cache_raw_columns(
        raw_columns, raw_data, working_columns, unit_handler, unit_kwargs, cache
    )

    data = {
        name: uuid_data[producer_uuid][name]
        for name, producer_uuid in working_columns.items()
        if producer_uuid in uuid_data and name in uuid_data[producer_uuid]
    }
    return data
