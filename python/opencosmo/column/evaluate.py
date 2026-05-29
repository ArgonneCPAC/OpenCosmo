from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from opencosmo.dataset.state import DatasetState


class EvaluateStrategy(Enum):
    VECTORIZE = "vectorize"
    ROW_WISE = "row_wise"
    CHUNKED = "chunked"


def evaluate_rows(
    data: dict[str, Any],
    func: Callable,
    kwargs: dict[str, Any],
    format: str,
):
    from opencosmo.dataset.formats import stack_rows

    data_length = len(next(iter(data.values())))
    per_column: dict[str, list] = {}
    for i in range(data_length):
        iterable_inputs = {name: values[i] for name, values in data.items()}
        output = func(**iterable_inputs, **kwargs)
        if not isinstance(output, dict):
            output = {func.__name__: output}
        for name, value in output.items():
            per_column.setdefault(name, []).append(value)
    return {name: stack_rows(values, format) for name, values in per_column.items()}


def evaluate_chunks(
    data: dict[str, Any],
    func: Callable,
    kwargs: dict[str, Any],
    chunk_sizes: np.ndarray,
    format: str,
):
    from opencosmo.dataset.formats import concat_chunks

    chunk_splits = np.cumsum(chunk_sizes)
    starts = np.concatenate([[0], chunk_splits[:-1]])
    per_column: dict[str, list] = {}
    for start, end in zip(starts, chunk_splits):
        chunk_input_data = {
            name: arr[int(start) : int(end)] for name, arr in data.items()
        }
        output = func(**chunk_input_data, **kwargs)
        if not isinstance(output, dict):
            output = {func.__name__: output}
        for name, value in output.items():
            per_column.setdefault(name, []).append(value)
    return {name: concat_chunks(chunks, format) for name, chunks in per_column.items()}


def evaluate_vectorized(data, func, kwargs, index):
    try:
        return func(**data, **kwargs, index=index)
    except TypeError:
        return func(**data, **kwargs)


def do_first_evaluation(
    func: Callable,
    strategy: str,
    format: str,
    kwargs: dict[str, Any],
    dataset: DatasetState,
):
    import opencosmo.dataset.state as st
    from opencosmo.dataset.formats import fetch_as_dict

    eval_strategy = EvaluateStrategy(strategy)
    columns = list(dataset.columns)
    match eval_strategy:
        case EvaluateStrategy.VECTORIZE:
            values = fetch_as_dict(st.take(dataset, 1), columns, format, unpack=False)
            return func(**values, **kwargs), eval_strategy

        case EvaluateStrategy.ROW_WISE:
            values = fetch_as_dict(st.take(dataset, 1), columns, format, unpack=False)
            values = {name: container[0] for name, container in values.items()}
            return func(**values, **kwargs), eval_strategy

        case EvaluateStrategy.CHUNKED:
            index = dataset.raw_index
            assert isinstance(index, tuple)
            first_chunk_size = index[1][0]
            first_chunk = fetch_as_dict(
                st.take(dataset, first_chunk_size, at="start"), columns, format
            )
            return func(**first_chunk, **kwargs), eval_strategy
