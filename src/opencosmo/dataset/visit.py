from inspect import signature
from typing import TYPE_CHECKING, Callable

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
):
    __verify(function, dataset)
    dataset = __prepare(function, dataset)
    if vectorize:
        result = __visit_vectorize(function, dataset, format)
        if not isinstance(result, dict):
            return {function.__name__: result}
        return result
    else:
        return __visit_rows(function, dataset, format)


def __visit_rows(function: Callable, dataset: "Dataset", format="astropy"):
    using_all_columns = (
        len(dataset.columns) > 1 and len(signature(function).parameters) == 1
    )
    storage = __make_output(function, dataset, using_all_columns)
    for i, row in enumerate(dataset.rows(output=format)):
        if using_all_columns:
            output = function(row)
        else:
            output = function(**row)
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


def __visit_vectorize(function: Callable, dataset: "Dataset", format: str = "astropy"):
    data = dataset.get_data(format)
    if format == "astropy" and isinstance(data, Table):
        data = {col.name: col.quantity for col in data.itercols()}
    elif isinstance(data, Column):
        data = {data.name: data.quantity}

    if not isinstance(data, dict) or (
        len(data) > 1 and len(signature(function).parameters) == 1
    ):
        return function(data)
    return function(**data)


def __prepare(function: Callable, dataset: "Dataset"):
    input_columns = signature(function).parameters.keys()
    if len(input_columns) == 1 and next(iter(input_columns)) == dataset.dtype:
        return dataset
    return dataset.select(input_columns)


def __verify(function: Callable, dataset: "Dataset"):
    input_columns = set(signature(function).parameters.keys())
    dataset_columns = set(dataset.columns)
    if not input_columns.issubset(dataset_columns):
        if len(input_columns) != 1 or input_columns.pop() != dataset.dtype:
            raise ValueError(
                f"Requested columns {input_columns - dataset_columns} not found in dataset"
            )
