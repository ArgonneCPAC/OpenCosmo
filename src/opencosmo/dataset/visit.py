from inspect import signature
from typing import TYPE_CHECKING, Any, Callable, Iterable

import astropy.units as u  # type: ignore
import numpy as np
from astropy.table import Column, Table  # type: ignore

from opencosmo.visit import insert

if TYPE_CHECKING:
    from opencosmo import Dataset


def visit_dataset(
    function: Callable,
    dataset: "Dataset",
    vectorize: bool = False,
    format: str = "astropy",
    evaluator_kwargs: dict[str, Any] = {},
):
    __verify(function, dataset, evaluator_kwargs.keys())
    dataset = __prepare(function, dataset, evaluator_kwargs.keys())
    if vectorize:
        result = __visit_vectorize(function, dataset, format, evaluator_kwargs)
        if not isinstance(result, dict):
            return {function.__name__: result}
        return result
    else:
        return __visit_rows(function, dataset, format, evaluator_kwargs)


def __visit_rows(
    function: Callable,
    dataset: "Dataset",
    format="astropy",
    evaluator_kwargs: dict[str, Any] = {},
):
    using_all_columns = (
        len(dataset.columns) > 1 and len(signature(function).parameters) == 1
    )
    storage = __make_output(function, dataset, using_all_columns)
    for i, row in enumerate(dataset.rows(output=format)):
        if using_all_columns:
            output = function(row, **evaluator_kwargs)
        else:
            output = function(**row, **evaluator_kwargs)
        insert(storage, i, output)
    return storage


def __make_output(function: Callable, dataset: "Dataset", using_all_columns: bool):
    if using_all_columns:
        first_values = function(next(dataset.take(1, at="start").rows()))
    else:
        first_values = function(**next(dataset.take(1, at="start").rows()))
    n = len(dataset)
    if not isinstance(first_values, dict):
        name = function.__name__
        first_values = {name: first_values}
    storage = {}
    for name, value in first_values.items():
        shape = (n,)
        dtype = type(value)
        if isinstance(value, np.ndarray):
            shape = shape + value.shape
            dtype = value.dtype
        storage[name] = np.zeros(shape, dtype=dtype)
    for name, value in first_values.items():
        if isinstance(value, u.Quantity):
            storage[name] = storage[name] * value.unit
    return storage


def __visit_vectorize(
    function: Callable,
    dataset: "Dataset",
    format: str = "astropy",
    evaluator_kwargs: dict[str, Any] = {},
):
    data = dataset.get_data(format)
    if format == "astropy" and isinstance(data, Table):
        data = {col.name: col.quantity for col in data.itercols()}
    elif isinstance(data, Column):
        data = {data.name: data.quantity}

    if not isinstance(data, dict) or (
        len(data) > 1 and len(signature(function).parameters) == 1
    ):
        return function(data, **evaluator_kwargs)
    return function(**data, **evaluator_kwargs)


def __prepare(function: Callable, dataset: "Dataset", evaluator_kwargs: Iterable[str]):
    input_columns = set(signature(function).parameters.keys())
    input_columns = input_columns.intersection(dataset.columns)
    if len(input_columns) == 1 and next(iter(input_columns)) == dataset.dtype:
        return dataset
    return dataset.select(input_columns)


def __verify(function: Callable, dataset: "Dataset", kwarg_names: Iterable[str]):
    input_names = set(signature(function).parameters.keys())
    dataset_columns = set(dataset.columns)
    kwarg_names = set(kwarg_names)
    missing = input_names - dataset_columns - kwarg_names
    if not missing:
        return
    elif len(missing) > 1 or next(iter(missing)) != dataset.dtype:
        raise ValueError(
            f"All inputs to the function must either be column names or passed as keyword arguments! Found unknown input(s) {','.join(missing)}"
        )
