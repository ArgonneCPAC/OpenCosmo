import rustworkx as rx

from opencosmo.column.column import RawColumn
from opencosmo.dataset.graph import build_dependency_graph, contract_derived_columns


def instantiate_dataset(
    column_producers, raw_data_handler, cache, unit_handler, unit_kwargs
):
    dependency_graph = build_dependency_graph(column_producers)
    cached_data = cache.get_data(dependency_graph.nodes())
    converted_cached_data = unit_handler.apply_unit_conversions(
        cached_data, unit_kwargs
    )

    push_up = True
    if converted_cached_data:
        push_up = False
        cache.add_data(converted_cached_data, {}, push_up=push_up)

    raw_columns = filter(
        lambda col: (
            isinstance(col, RawColumn)
            and not col.requires.intersection(cached_data.keys())
        ),
        column_producers,
    )
    raw_data = raw_data_handler.get_data([col.name for col in raw_columns])
    raw_data = unit_handler.apply_raw_units(raw_data, unit_kwargs)
    new_derived_columns = build_derived_columns(
        column_producers, converted_cached_data | raw_data, dependency_graph, None
    )
    if raw_data:
        cache.add_data(raw_data, {}, push_up=True)
    converted_raw_data = unit_handler.apply_unit_conversions(raw_data, unit_kwargs)
    if converted_raw_data:
        cache.add_data(converted_raw_data, {}, push_up=False)
        raw_data |= converted_raw_data

    if new_derived_columns:
        cache.add_data(new_derived_columns, {}, push_up=push_up)

    return converted_cached_data | raw_data | new_derived_columns


def build_derived_columns(column_producers, data, dependency_graph, index):
    dependency_graph = contract_derived_columns(dependency_graph, column_producers)
    new_derived = {}
    for colidx in rx.topological_sort(dependency_graph):
        column = dependency_graph[colidx]
        if isinstance(column, str):
            assert column in data
            continue
        produces = column.produces
        if all(name in data for name in produces):
            continue
        output = column.evaluate(data, index[1] if isinstance(index, tuple) else None)
        if isinstance(output, dict):
            new_derived |= output
        else:
            new_derived[column.name] = output
    return new_derived
