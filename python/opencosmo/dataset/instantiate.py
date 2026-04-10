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


def instantiate_dataset(
    column_producers,
    column_names,
    raw_data_handler,
    cache,
    unit_handler,
    unit_kwargs,
):
    dependency_graph = build_dependency_graph(column_producers)
    all_required_columns = get_all_required_columns(column_names, dependency_graph)
    cached_data = cache.get_data(all_required_columns)
    converted_cached_data = unit_handler.apply_unit_conversions(
        cached_data, unit_kwargs
    )

    if converted_cached_data:
        cache.add_data(converted_cached_data, {}, push_up=False)
        cached_data |= converted_cached_data

    raw_columns = list(
        filter(
            lambda col: (
                isinstance(col, RawColumn)
                and col.name not in cached_data
                and col.name in all_required_columns
            ),
            column_producers,
        )
    )
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

    return cached_data | raw_data | new_derived_columns


def build_derived_columns(
    column_producers, column_names, data, dependency_graph, index
):
    dependency_graph = contract_derived_columns(
        dependency_graph, column_names, column_producers
    )
    new_derived = {}
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
