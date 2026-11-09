from __future__ import annotations

from dataclasses import replace
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any

from astropy.units import Quantity

from opencosmo import facade
from opencosmo.collection.lightcone import operations as lcops
from opencosmo.collection.lightcone.state import LightconeState
from opencosmo.collection.structure import operations as scops
from opencosmo.collection.structure.state import StructureState, public_keys
from opencosmo.column.evaluate import function_name
from opencosmo.column.select import (
    MissingColumnError,
    build_multi_dataset_selections,
)
from opencosmo.dataset import operations as dsops
from opencosmo.dataset.formats import concat_chunks, stack_rows
from opencosmo.dataset.state import DatasetState

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping


def verify_evaluate_on_collection(
    function: Callable,
    state: StructureState,
    evaluate_kwargs: dict[str, Any],
    dataset: str | None,
):
    # Case 1/3
    function_signature = signature(function)
    function_arg_names = set(function_signature.parameters.keys())
    datasets_in_collection = set(public_keys(state))
    if not (
        requested_datasets := function_arg_names.intersection(datasets_in_collection)
    ):
        raise ValueError(
            "Your function should take the names of some of the datasets in this collection as arguments!"
        )
    elif dataset is not None and dataset not in requested_datasets:
        raise ValueError(
            "If you pass an argument to 'dataset', your function must take in at least one column from that dataset"
        )

    required_parameters = {
        name
        for name, par in function_signature.parameters.items()
        if par.default == Parameter.empty
    }
    if missing := required_parameters.difference(datasets_in_collection).difference(
        evaluate_kwargs.keys()
    ):
        raise ValueError(
            f"Your function has required arguments {missing}, but you didn't provide them!"
        )

    spec = {name: evaluate_kwargs.pop(name, None) for name in requested_datasets}
    return spec, evaluate_kwargs


def resolve_column_arguments(
    function: Callable,
    columns_by_dataset: Mapping[tuple[str, ...], Iterable[str]],
    evaluate_kwargs: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """
    Map each parameter of ``function`` that names a column to the path of the
    one dataset that contains it. Parameters that match no column must be
    supplied through ``evaluate_kwargs`` or have defaults. A parameter that is
    both a column and an evaluate kwarg is ambiguous and raises.
    """
    columns_by_dataset = {
        path: set(columns) for path, columns in sorted(columns_by_dataset.items())
    }
    parameters = {
        name: parameter
        for name, parameter in signature(function).parameters.items()
        if parameter.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
    }
    # Lengths only route array kwargs, which are never passed here
    names_by_dataset, _, unmatched = build_multi_dataset_selections(
        columns_by_dataset,
        dict.fromkeys(columns_by_dataset, 0),
        (list(parameters),),
        {},
    )

    paths_by_name: dict[str, list[tuple[str, ...]]] = {}
    for path, names in names_by_dataset.items():
        for name in names:
            paths_by_name.setdefault(name, []).append(path)
    for name, paths in paths_by_name.items():
        found_in = ", ".join(".".join(path) for path in paths)
        if len(paths) > 1:
            raise ValueError(
                f"Column {name} was found in multiple datasets ({found_in}). "
                "Use with_datasets to narrow the collection, or take dataset names "
                "as arguments to your function."
            )
        if name in evaluate_kwargs:
            raise ValueError(
                f"{name} was passed to evaluate as a keyword argument, but it is "
                f"also a column in {found_in}. Rename the function parameter to "
                "pass a value that is not a column."
            )

    missing = [
        name
        for name in unmatched
        if name not in evaluate_kwargs and parameters[name].default is Parameter.empty
    ]
    if missing:
        raise MissingColumnError(
            f"Your function has required arguments {missing}, but they are not "
            "columns in any dataset in this collection and you didn't provide them!"
        )
    return {name: paths[0] for name, paths in paths_by_name.items()}


def visit_structure_eagerly(
    function: Callable,
    state: StructureState,
    format: str = "astropy",
    dataset: str | None = None,
    evaluate_kwargs: dict[str, Any] | None = None,
    insert: bool = True,
):
    if evaluate_kwargs is None:
        evaluate_kwargs = {}
    spec, kwargs = verify_evaluate_on_collection(
        function, state, evaluate_kwargs, dataset
    )

    to_visit = __prepare_state(spec, state)

    if dataset is None:
        inputs = (
            __make_input(structure, format)
            for structure in scops.iter_objects(to_visit)
        )
        return evaluate_into_properties(function, inputs, format, kwargs, insert)
    else:
        return evaluate_into_dataset(
            function, to_visit, format, kwargs, dataset, insert
        )


def visit_structure_by_columns(
    function: Callable,
    state: StructureState,
    columns: Mapping[str, tuple[str, ...]],
    format: str,
    evaluate_kwargs: dict[str, Any],
    insert: bool,
):
    """
    Evaluate ``function`` once per structure, passing each column argument
    the values from the dataset ``columns`` maps it to.
    """
    by_path: dict[tuple[str, ...], list[str]] = {}
    for name, path in sorted(columns.items()):
        by_path.setdefault(path, []).append(name)
    to_visit = __restrict_to_columns(state, by_path)
    inputs = __column_inputs(to_visit, by_path, format)
    return evaluate_into_properties(function, inputs, format, evaluate_kwargs, insert)


def evaluate_into_properties(
    function: Callable,
    inputs: Iterable[Mapping[str, Any]],
    format: str,
    kwargs: dict[str, Any],
    insert: bool,
):
    per_column: dict[str, list] = {}
    for input_structure in inputs:
        output = function(**input_structure, **kwargs)
        if output is None and insert:
            raise ValueError(
                "You asked to insert these values, but your function returns None!"
            )
        if not isinstance(output, dict):
            output = {function_name(function): output}
        for name, value in output.items():
            per_column.setdefault(name, []).append(value)

    if not per_column:
        return None
    return {name: stack_rows(values, format) for name, values in per_column.items()}


def evaluate_into_dataset(
    function: Callable,
    state: StructureState,
    format: str,
    kwargs: dict[str, Any],
    dataset: str,
    insert: bool,
):
    per_column: dict[str, list] = {}
    for i, structure in enumerate(scops.iter_objects(state)):
        expected_length = len(structure[dataset])
        input_structure = __make_input(structure, format)
        output = function(**input_structure, **kwargs)
        if output is None and insert:
            raise ValueError(
                "You asked to insert these values, but your function returns None!"
            )
        if not isinstance(output, dict):
            output = {function_name(function): output}

        if i == 0 and any(len(v) != expected_length for v in output.values()):
            raise ValueError(
                "If you pass a `dataset` argument, your function should output an array with the same length as that dataset"
            )
        for name, output_arr in output.items():
            per_column.setdefault(name, []).append(output_arr)

    if not per_column:
        return None
    return {name: concat_chunks(data, format) for name, data in per_column.items()}


def __make_input(structure: Mapping[str, Any], format: str = "astropy"):
    values = {}
    for name, element in structure.items():
        if isinstance(element, dict):
            values[name] = __make_input(element, format)
        elif isinstance(element, DatasetState):
            values[name] = dsops.get_data(
                element, format, unpack=False, wrap_single=True
            )
        elif isinstance(element, (StructureState, LightconeState)):
            values[name] = facade.wrap(element)
        elif isinstance(element, Quantity) and format != "astropy":
            values[name] = element.value
        else:
            values[name] = element
    return values


def __prepare_state(
    spec: dict[str, list[str] | None], state: StructureState
) -> StructureState:
    state = scops.with_datasets(state, list(spec.keys()))
    selections = {ds_name: cols for ds_name, cols in spec.items() if cols is not None}
    return scops.select_with(state, (), selections)


def __restrict_to_columns(
    state: StructureState, by_path: Mapping[tuple[str, ...], list[str]]
) -> StructureState:
    by_child: dict[str, dict[tuple[str, ...], list[str]]] = {}
    for path, names in by_path.items():
        by_child.setdefault(path[0], {})[path[1:]] = names
    # Keep the source visible so iteration yields every structure, including
    # those with no rows in the requested linked datasets.
    source_columns = by_child.pop(state.dtype, {}).get(
        (), [next(iter(state.source.columns))]
    )
    state = replace(
        state,
        linked={
            name: child for name, child in state.linked.items() if name in by_child
        },
        hide_source=False,
    )
    selections = {name: sub[()] for name, sub in by_child.items() if () in sub}
    state = scops.select_with(state, (), {state.dtype: source_columns, **selections})
    nested: dict[str, StructureState] = {}
    for name, sub in by_child.items():
        if () in sub:
            continue
        child = state.linked[name]
        assert isinstance(child, StructureState)
        nested[name] = __restrict_to_columns(child, sub)
    return replace(state, linked={**state.linked, **nested})


def __column_inputs(
    state: StructureState,
    by_path: Mapping[tuple[str, ...], list[str]],
    format: str,
) -> Generator[dict[str, Any]]:
    leaves = scops.leaf_datasets(state)
    empty: dict[tuple[str, ...], Any] = {}
    for structure in scops.iter_objects(state):
        values = {}
        for path, names in by_path.items():
            element = structure.get(path[0])
            if isinstance(element, StructureState):
                element = scops.leaf_datasets(element)[path[1:]]
            if element is None:
                if path not in empty:
                    empty[path] = __leaf_data(__empty_leaf(leaves[path]), format)
                data = empty[path]
            else:
                data = __leaf_data(element, format)
            values.update({name: data[name] for name in names})
        yield values


def __leaf_data(element: Any, format: str) -> Any:
    if isinstance(element, dict):
        return __make_input(element, format)
    if isinstance(element, DatasetState):
        return dsops.get_data(element, format, unpack=False, wrap_single=True)
    return lcops.get_data(element, format, unpack=False, wrap_single=True)


def __empty_leaf(
    leaf: DatasetState | LightconeState,
) -> DatasetState | LightconeState:
    if isinstance(leaf, DatasetState):
        return dsops.take_range(leaf, 0, 0, "local")
    return lcops.take_range(leaf, 0, 0, "local")
