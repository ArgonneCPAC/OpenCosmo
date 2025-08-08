from inspect import signature
from typing import TYPE_CHECKING, Callable

import astropy.units as u
import numpy as np

from opencosmo.visit import insert

if TYPE_CHECKING:
    from opencosmo import Dataset


def visit_dataset(function: Callable, dataset: "Dataset", vectorize: bool = False):
    __verify(function, dataset)
    dataset = __prepare(function, dataset)
    if vectorize:
        print("hi")
        result = __visit_vectorize(function, dataset)
        if not isinstance(result, dict):
            return {function.__name__: result}
        return result
    else:
        return __visit_rows(function, dataset)


def __visit_rows(function: Callable, dataset: "Dataset"):
    storage = __make_output(function, dataset)
    for i, row in enumerate(dataset.rows()):
        output = function(**row)
        insert(storage, i, output)
    return storage


def __make_output(function: Callable, dataset: "Dataset"):
    first_values = function(**next(dataset.take(1, at="start").rows()))
    n = len(dataset)
    if not isinstance(first_values, dict):
        name = function.__name__
        first_values = {name: first_values}
    storage = {name: np.zeros(n, dtype=type(val)) for name, val in first_values.items()}
    for name, value in first_values.items():
        if isinstance(value, u.Quantity):
            storage[name] = storage[name] * value.unit
    return storage


def __visit_vectorize(function: Callable, dataset: "Dataset"):
    data = dataset.get_data("numpy")
    if not isinstance(data, dict):
        return function(data)
    return function(**data)


def __prepare(function: Callable, dataset: "Dataset"):
    input_columns = signature(function).parameters.keys()
    return dataset.select(input_columns)


def __verify(function: Callable, dataset: "Dataset"):
    input_columns = set(signature(function).parameters.keys())
    dataset_columns = set(dataset.columns)
    if not input_columns.issubset(dataset_columns):
        raise ValueError(
            f"Requested columns {input_columns - dataset_columns} not found in dataset"
        )
