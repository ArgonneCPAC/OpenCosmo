from __future__ import annotations

from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Callable, Optional

from astropy.units import Quantity  # type: ignore

from opencosmo import dataset as ds
from opencosmo.dataset.formats import concat_chunks, stack_rows

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
    per_column: dict[str, list] = {}
    for structure in collection.objects():
        input_structure = __make_input(structure, format)
        output = function(**input_structure, **kwargs)
        if output is None and insert:
            raise ValueError(
                "You asked to insert these values, but your function returns None!"
            )
        if not isinstance(output, dict):
            output = {function.__name__: output}
        for name, value in output.items():
            per_column.setdefault(name, []).append(value)

    if not per_column:
        return None
    return {name: stack_rows(values, format) for name, values in per_column.items()}


def evaluate_into_dataset(
    function: Callable,
    collection: StructureCollection,
    format: str,
    kwargs: dict[str, Any],
    dataset: str,
    insert: bool,
):
    per_column: dict[str, list] = {}
    for i, structure in enumerate(collection.objects()):
        expected_length = len(structure[dataset])
        input_structure = __make_input(structure, format)
        output = function(**input_structure, **kwargs)
        if output is None and insert:
            raise ValueError(
                "You asked to insert these values, but your function returns None!"
            )
        if not isinstance(output, dict):
            output = {function.__name__: output}

        if i == 0:
            if any(len(v) != expected_length for v in output.values()):
                raise ValueError(
                    "If you pass a `dataset` argument, your function should output an array with the same length as that dataset"
                )
        for name, output_arr in output.items():
            per_column.setdefault(name, []).append(output_arr)

    if not per_column:
        return None
    return {name: concat_chunks(data, format) for name, data in per_column.items()}


def __make_input(structure: dict, format: str = "astropy"):
    values = {}
    for name, element in structure.items():
        if isinstance(element, dict):
            values[name] = __make_input(element, format)
        elif isinstance(element, ds.Dataset):
            data = element.get_data(format, wrap_single=True, unpack=False)
            values[name] = data
        elif isinstance(element, Quantity) and format != "astropy":
            values[name] = element.value
        else:
            values[name] = element
    return values


def __prepare_collection(
    spec: dict[str, Optional[list[str]]], collection: StructureCollection
) -> StructureCollection:
    collection = collection.with_datasets(list(spec.keys()))
    selections = {ds_name: cols for ds_name, cols in spec.items() if cols is not None}
    collection = collection.select(**selections)
    return collection
