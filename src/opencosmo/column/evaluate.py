from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import astropy.units as u
import numpy as np

from opencosmo.evaluate import insert_data, prepare_kwargs

if TYPE_CHECKING:
    from opencosmo import Dataset
    from opencosmo.index import ChunkedIndex


class EvaluateStrategy(Enum):
    VECTORIZE = "vectorize"
    ROW_WISE = "row_wise"
    CHUNKED = "chunked"


def evaluate_rows(data: dict[str, np.ndarray], func: Callable, kwargs: dict[str, Any]):
    data_length = len(next(iter(data.values())))
    kwargs_, iterable_kwargs = prepare_kwargs(data_length, kwargs)
    iterable_data = data | iterable_kwargs
    storage = {}
    for i in range(data_length):
        iterable_inputs = {name: values[i] for name, values in iterable_data.items()}
        output = func(**iterable_inputs, **kwargs_)
        if not isinstance(output, dict):
            output = {func.__name__: output}
        if i == 0:
            storage = __make_row_based_output_from_first_values(output, data_length)
            continue
        insert_data(storage, i, output)
    return storage


def __make_row_based_output_from_first_values(values, data_length):
    storage = {}
    for name, value in values.items():
        try:
            shape = (data_length,) + value.shape
        except AttributeError:
            shape = (data_length,)
        try:
            dtype = value.dtype
        except AttributeError:
            dtype = type(value)
        column_storage = np.zeros(shape, dtype=dtype)
        if isinstance(value, u.Quantity):
            column_storage *= value.unit
        column_storage[0] = value
        storage[name] = column_storage

    return storage


def evaluate_chunks(
    data: dict[str, np.ndarray],
    func: Callable,
    kwargs: dict[str, Any],
    index: ChunkedIndex,
):
    data_length = len(next(iter(data.values())))
    kwargs_, iterable_kwargs = prepare_kwargs(data_length, kwargs)
    input_data = data | iterable_kwargs

    chunk_splits = np.cumsum(index[1])
    storage = {}
    input_data = {name: np.split(arr, chunk_splits) for name, arr in data.items()}
    for i in range(len(chunk_splits)):
        chunk_input_data = {name: split[i] for name, split in input_data.items()}
        output = func(**chunk_input_data, **kwargs_)
        if not isinstance(output, dict):
            output = {func.__name__: output}
        if i == 0:
            storage = __make_chunked_based_output_from_first_values(output, data_length)
            continue
        for name, values in output.items():
            storage[name][chunk_splits[i - 1] : chunk_splits[i]] = values
    return storage


def __make_chunked_based_output_from_first_values(values, data_length):
    storage = {}
    for name, value in values.items():
        shape = (data_length,) + value.shape[1:]
        dtype = value.dtype
        column_storage = np.zeros(shape, dtype=dtype)
        if isinstance(value, u.Quantity):
            column_storage *= value.unit
        column_storage[0 : len(value)] = value
        storage[name] = column_storage

    return storage


def evaluate_vectorized(data, func, kwargs):
    return func(**data, **kwargs)


def do_first_evaluation(
    func: Callable, strategy: str, format: str, kwargs: dict[str, Any], dataset: Dataset
):
    eval_strategy = EvaluateStrategy(strategy)
    match eval_strategy:
        case EvaluateStrategy.VECTORIZE:
            values = dataset.take(1).get_data(format, unpack=False)
            try:
                values = dict(values)
            except TypeError:
                values = {dataset.columns[0]: values}

            return func(**values, **kwargs), eval_strategy

        case EvaluateStrategy.ROW_WISE:
            values = dataset.take(1).get_data(format, unpack=True)
            try:
                values = dict(values)
            except TypeError:
                values = {dataset.columns[0]: values}
            return func(**values, **kwargs), eval_strategy

        case EvaluateStrategy.CHUNKED:
            index = dataset.index
            assert isinstance(index, tuple)
            first_chunk_size = index[1][0]
            first_chunk = dataset.take(first_chunk_size, at="start").get_data(format)
            first_chunk = dict(first_chunk)
            return func(**first_chunk, **kwargs), eval_strategy
