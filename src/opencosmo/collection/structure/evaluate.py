from __future__ import annotations

from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np
from astropy.units import Quantity  # type: ignore

from opencosmo import dataset as ds
from opencosmo.evaluate import (
    insert_data,
    make_output_from_first_values,
    prepare_kwargs,
)

if TYPE_CHECKING:
    from opencosmo import StructureCollection


def verify_evaluate_on_collection(
    function: Callable,
    collection: StructureCollection,
    evaluate_kwargs: dict[str, Any],
    dataset: Optional[str],
):
    # Case 1/3
    function_signature = signature(function)
    function_arg_names = set(function_signature.parameters.keys())
    datasets_in_collection = set(collection.keys())
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


def visit_structure_collection_eagerly(
    function: Callable,
    collection: StructureCollection,
    format: str = "astropy",
    dataset: Optional[str] = None,
    evaluate_kwargs: dict[str, Any] = {},
    insert: bool = True,
):
    spec, kwargs = verify_evaluate_on_collection(
        function, collection, evaluate_kwargs, dataset
    )

    to_visit = __prepare_collection(spec, collection)

    if dataset is None:
        return evaluate_into_properties(function, to_visit, format, kwargs, insert)
    else:
        return evaluate_into_dataset(
            function, to_visit, format, kwargs, dataset, insert
        )


def evaluate_into_properties(
    function: Callable,
    collection: StructureCollection,
    format: str,
    kwargs: dict[str, Any],
    insert: bool,
):
    kwargs, iterable_kwargs = prepare_kwargs(len(collection), kwargs)

    storage = __make_output(
        function, collection, format, kwargs, iterable_kwargs, insert
    )
    for i, structure in enumerate(collection.objects()):
        if i == 0:
            continue
        iterable_kwarg_values = {name: arr[i] for name, arr in iterable_kwargs.items()}
        input_structure = __make_input(structure, format)

        output = function(**input_structure, **kwargs, **iterable_kwarg_values)
        if storage is not None:
            insert_data(storage, i, output)

    return storage


def evaluate_into_dataset(
    function: Callable,
    collection: StructureCollection,
    format: str,
    kwargs: dict[str, Any],
    dataset: str,
    insert: bool,
):
    kwargs, iterable_kwargs = prepare_kwargs(len(collection[dataset]), kwargs)
    storage = __make_chunked_output(
        function, collection, dataset, format, kwargs, iterable_kwargs
    )

    for i, structure in enumerate(collection.objects()):
        if i == 0:
            continue
        iterable_kwarg_values = {name: arr[i] for name, arr in iterable_kwargs.items()}
        input_structure = __make_input(structure, format)

        output = function(**input_structure, **kwargs, **iterable_kwarg_values)
        if storage is not None:
            for name, output_arr in output:
                storage[name].append(output_arr)

    if storage is None:
        return
    output_data = {name: np.concatenate(data) for name, data in storage.items()}
    return output_data


def __make_input(structure: dict, format: str = "astropy"):
    values = {}
    for name, element in structure.items():
        if isinstance(element, dict):
            values[name] = __make_input(element, format)
        elif isinstance(element, ds.Dataset):
            data = element.get_data(format)
            values[name] = data
        elif isinstance(element, Quantity) and format == "numpy":
            values[name] = element.value
        else:
            values[name] = element
    return values


def __make_output(
    function: Callable,
    collection: StructureCollection,
    format: str = "astropy",
    kwargs: dict[str, Any] = {},
    iterable_kwargs: dict[str, Sequence] = {},
    insert: bool = True,
) -> dict | None:
    first_structure = next(collection.take(1, at="start").objects())
    first_input = __make_input(first_structure, format)
    first_values = function(
        **first_input,
        **kwargs,
        **{name: arr[0] for name, arr in iterable_kwargs.items()},
    )
    if first_values is None and insert:
        raise ValueError(
            "You asked to insert these values, but your function returns None!"
        )
    elif first_values is None:
        return None
    if not isinstance(first_values, dict):
        name = function.__name__
        first_values = {name: first_values}
    n_rows = len(collection)
    return make_output_from_first_values(first_values, n_rows)


def __make_chunked_output(
    function: Callable,
    collection: StructureCollection,
    dataset: str,
    format: str = "astropy",
    kwargs: dict[str, Any] = {},
    iterable_kwargs: dict[str, Sequence] = {},
    insert: bool = True,
) -> dict | None:
    first_structure = collection.take(1, at="start").objects()
    expected_length = len(first_structure[dataset])
    first_structure_data = next(iter(first_structure.objects()))

    first_input = __make_input(first_structure_data, format)
    first_values = function(
        **first_input,
        **kwargs,
        **{name: arr[0] for name, arr in iterable_kwargs.items()},
    )
    if first_values is None and insert:
        raise ValueError(
            "You asked to insert these values, but your function returns None!"
        )
    elif first_values is None:
        return None
    if not isinstance(first_values, dict):
        name = function.__name__
        first_values = {name: first_values}
    if any(len(fv) != expected_length for fv in first_values.values()):
        raise ValueError(
            "If you pass a `dataset` argument, your function should output an array with the same length as that dataset"
        )
    return {name: [fv] for name, fv in first_values.items()}


def __prepare_collection(
    spec: dict[str, Optional[list[str]]], collection: StructureCollection
) -> StructureCollection:
    collection = collection.with_datasets(list(spec.keys()))
    selections = {ds_name: cols for ds_name, cols in spec.items() if cols is not None}
    collection = collection.select(**selections)
    return collection
