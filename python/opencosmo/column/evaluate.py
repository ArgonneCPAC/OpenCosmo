from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np

if TYPE_CHECKING:
    from opencosmo.dataset.state import DatasetState


class EvaluateStrategy(Enum):
    VECTORIZE = "vectorize"
    ROW_WISE = "row_wise"
    CHUNKED = "chunked"


def evaluate_chunks(
    data: dict[str, Any],
    func: Callable,
    kwargs: dict[str, Any],
    chunk_sizes: int | np.ndarray,
    format: str,
    should_unpack_data: bool,
):
    from opencosmo.dataset.formats import concat_chunks, stack_rows

    ranges: Iterable[tuple[int | np.integer, int | np.integer]]
    if isinstance(chunk_sizes, int):
        if chunk_sizes <= 0:
            raise ValueError("Chunk size must be positive")
        data_length = len(next(iter(data.values())))
        ranges = (
            (start, min(start + chunk_sizes, data_length))
            for start in range(0, data_length, chunk_sizes)
        )
    else:
        chunk_splits = np.cumsum(chunk_sizes)
        starts = np.concatenate([[0], chunk_splits[:-1]])
        ranges = zip(starts, chunk_splits)

    per_column: dict[str, list] = {}
    for start, end in ranges:
        chunk_input_data = {
            name: arr[int(start) : int(end)] for name, arr in data.items()
        }
        if should_unpack_data:
            output = func(**chunk_input_data, **kwargs)
        else:
            output = func(data=chunk_input_data, **kwargs)
        if not isinstance(output, dict):
            output = {func.__name__: output}
        for name, value in output.items():
            per_column.setdefault(name, []).append(value)

    output = {}
    for name, column_chunks in per_column.items():
        try:
            output[name] = concat_chunks(column_chunks, format)
        except (TypeError, ValueError):
            output[name] = stack_rows(column_chunks, format)
    return output


def evaluate_vectorized(data, func, kwargs, index, should_unpack_data):
    try:
        if should_unpack_data:
            return func(**data, **kwargs, index=index)
        return func(data=data, **kwargs, index=index)
    except TypeError:
        if should_unpack_data:
            return func(**data, **kwargs)
        return func(data=data, **kwargs)


def do_first_evaluation(
    func: Callable,
    strategy: str,
    format: str,
    kwargs: dict[str, Any],
    state: DatasetState,
    should_unpack_data: bool,
):
    from opencosmo.dataset import operations as dsops
    from opencosmo.dataset.formats import fetch_as_dict

    eval_strategy = EvaluateStrategy(strategy)
    columns = list(state.columns)
    match eval_strategy:
        case EvaluateStrategy.VECTORIZE:
            values = fetch_as_dict(
                dsops.take(state, 1, "start", "local"), columns, format, unpack=False
            )
            if should_unpack_data:
                return func(**values, **kwargs), eval_strategy
            return func(data=values, **kwargs), eval_strategy

        case EvaluateStrategy.ROW_WISE:
            values = fetch_as_dict(
                dsops.take(state, 1, "start", "local"), columns, format, unpack=False
            )
            values = {name: container[0] for name, container in values.items()}
            if should_unpack_data:
                return func(**values, **kwargs), eval_strategy
            return func(data=values, **kwargs), eval_strategy

        case EvaluateStrategy.CHUNKED:
            index = state.raw_index
            assert isinstance(index, tuple)
            first_chunk_size = index[1][0]
            first_chunk = fetch_as_dict(
                dsops.take(state, first_chunk_size, at="start", mode="local"),
                columns,
                format,
                unpack=False,
            )
            if should_unpack_data:
                return func(**first_chunk, **kwargs), eval_strategy
            return func(data=first_chunk, **kwargs), eval_strategy
