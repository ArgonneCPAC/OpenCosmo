from __future__ import annotations

from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
from astropy.units import Quantity  # type: ignore

from opencosmo import dataset as ds
from opencosmo.dataset.evaluate import visit_dataset
from opencosmo.evaluate import insert, make_output_from_first_values, prepare_kwargs

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from opencosmo import StructureCollection


def visit_structure_collection(
    function: Callable,
    spec: Mapping[str, Optional[list[str]]],
    collection: StructureCollection,
    format: str = "astropy",
    dtype: Optional[DTypeLike] = None,
    evaluator_kwargs: dict[str, Any] = {},
):
    spec = dict(spec)
    __verify(function, spec, collection, evaluator_kwargs.keys())
    to_visit = __prepare_collection(spec, collection)
    kwargs, iterable_kwargs = prepare_kwargs(len(collection), evaluator_kwargs)
    if dtype is None:
        dtype = np.float64

    storage = __make_output(function, to_visit, format, kwargs, iterable_kwargs)
    for i, structure in enumerate(to_visit.objects()):
        if i == 0:
            continue
        iterable_kwarg_values = {name: arr[i] for name, arr in iterable_kwargs.items()}
        input_structure = __make_input(structure, format)

        output = function(**input_structure, **kwargs, **iterable_kwarg_values)
        if storage is not None:
            insert(storage, i, output)

    return storage


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
) -> dict | None:
    first_structure = next(collection.take(1, at="start").objects())
    first_input = __make_input(first_structure, format)
    first_values = function(
        **first_input,
        **kwargs,
        **{name: arr[0] for name, arr in iterable_kwargs.items()},
    )
    if first_values is None:
        return None
    if not isinstance(first_values, dict):
        name = function.__name__
        first_values = {name: first_values}
    n_rows = len(collection)
    return make_output_from_first_values(first_values, n_rows)


def __prepare_collection(
    spec: dict[str, Optional[list[str]]], collection: StructureCollection
) -> StructureCollection:
    collection = collection.with_datasets(list(spec.keys()))
    selections = {ds_name: cols for ds_name, cols in spec.items() if cols is not None}
    collection = collection.select(**selections)
    return collection


def __verify(
    function: Callable,
    spec: dict[str, Optional[list[str]]],
    collection: StructureCollection,
    kwarg_keys: Iterable[str],
):
    datasets_in_collection = set(collection.keys())
    kwarg_keys = set(kwarg_keys)
    fn_signature = signature(function)
    parameters = fn_signature.parameters

    for name, param in parameters.items():
        if name not in spec and name not in kwarg_keys and param == Parameter.empty:
            spec.update({name: None})

    datasets_in_spec = set(spec.keys())

    if not datasets_in_spec.issubset(datasets_in_collection):
        raise ValueError(
            "This collection is missing datasets "
            f"{datasets_in_spec - datasets_in_collection} requested for this visitor"
        )
    for ds_name, columns_in_spec in spec.items():
        if columns_in_spec is None:
            continue
        dataset = collection[ds_name]
        if not isinstance(dataset, ds.Dataset):
            if not isinstance(columns_in_spec, dict):
                raise ValueError(
                    "When passing columns to a nested structure collection, the argument should be a dictionary"
                )
                for key, value in columns_in_spec.items():
                    if key not in dataset.keys():
                        raise ValueError(
                            "No dataset {key} found in structure collection"
                        )
                    elif set(dataset[key].columns).difference(value):
                        raise ValueError(
                            "Missing some requested columns in this datset!"
                        )
            continue

        columns_to_check = set(columns_in_spec)
        columns_in_dataset = set(dataset.columns)
        if not columns_to_check.issubset(columns_in_dataset):
            raise ValueError(
                "Dataset {ds_name} is missing columns "
                f"{columns_to_check - columns_in_dataset} requested for this visitor"
            )

    if not datasets_in_spec.issubset(parameters.keys()):
        raise ValueError(
            "Visitor function must use the names of the datasets it requests as its "
            "argument names"
        )
