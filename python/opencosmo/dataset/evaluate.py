from __future__ import annotations

from collections import defaultdict
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np
from astropy.units import Quantity

from opencosmo.column.column import EvaluatedColumn
from opencosmo.column.evaluate import EvaluateStrategy, do_first_evaluation
from opencosmo.dataset import operations as dsops
from opencosmo.dataset.formats import concat_chunks, fetch_as_dict

if TYPE_CHECKING:
    from opencosmo.dataset.state import DatasetState

"""
Although the user-facing name for this operation is "evaluate", the pattern 
we are using here is known as a "visitor."
"""


def build_evaluated_column(
    dataset, func, vectorize, insert, format, batch_size, evaluate_kwargs
):
    kwarg_columns = set(evaluate_kwargs.keys()).intersection(dataset.columns)
    if kwarg_columns:
        raise ValueError(
            "Keyword arguments cannot have the same name as columns in your dataset!"
        )

    match (vectorize, batch_size):
        case (True, -1):
            default_strategy = "vectorize"
        case (False, -1):
            default_strategy = "row_wise"
        case (_, _):
            default_strategy = "vectorize"

    strategy = evaluate_kwargs.pop("strategy", default_strategy)
    # Structure collections pass the "chunked" strategy to datasets, which causes the dataset
    # To be evaluated on a structure-by-structure basis. This supersedes all other options.
    if strategy == "chunked":
        batch_size = -1

    return verify_for_lazy_evaluation(
        func,
        strategy,
        format,
        evaluate_kwargs,
        dataset,
        batch_size,
        skip_evaluation_check=not insert,
    )


def visit_dataset(
    column: EvaluatedColumn,
    state: DatasetState,
    batch_size: int,
) -> dict[str, np.ndarray]:
    if column.batch_size > 0:
        return visit_dataset_batched(column, state)
    data = fetch_as_dict(state, column.requires_names, column.format, unpack=False)
    output = column.evaluate(data, state.raw_index)
    if not isinstance(output, dict):
        assert len(column.produces) == 1
        output = {column.produces.pop(): output}
    return output


def visit_dataset_batched(column: EvaluatedColumn, state: DatasetState):
    length = len(state)
    ranges = np.arange(0, length, column.batch_size)
    if ranges[-1] != length:
        ranges = np.append(ranges, length)

    output = defaultdict(list)

    for start, end in np.lib.stride_tricks.sliding_window_view(ranges, 2):
        batch_data = fetch_as_dict(
            dsops.take_range(state, start, end, "local"),
            column.requires_names,
            column.format,
            unpack=False,
        )
        batch_output = column.evaluate(batch_data, None)
        if batch_output is not None and not isinstance(batch_output, dict):
            batch_output = {column.produces.pop(): batch_output}

        for name, column_batch in batch_output.items():
            output[name].append(column_batch)
    full_output = {
        name: concat_chunks(out, column.format) for name, out in output.items()
    }
    return full_output


def verify_for_lazy_evaluation(
    func: Callable,
    strategy: str,
    format: str,
    evaluator_kwargs: dict[str, Any],
    state: DatasetState,
    batch_size: int,
    allow_none=False,
    skip_evaluation_check=False,
) -> EvaluatedColumn:
    """
    Verify the function behaves correctly and determine the names of its output columns.
    """
    __verify(func, state.columns, evaluator_kwargs.keys())
    sig = signature(func)
    parameter_names = set(sig.parameters.keys())
    required_parameters = {
        name
        for name, param in sig.parameters.items()
        if param.default == Parameter.empty
    }

    # Required parameters that correspond to dataset columns.
    column_arguments = required_parameters.intersection(state.columns)
    # Required parameters that are not satisfied by evaluator kwargs.
    required_not_columns = required_parameters.difference(state.columns)
    required_not_columns = required_not_columns.difference(evaluator_kwargs.keys())

    # Validate that any remaining required (non-kwarg) inputs are either `data` (data-mapping
    # mode) or empty (column-unpacking mode).
    unknown_required = required_not_columns.difference({"data"})
    if unknown_required:
        raise ValueError(
            f"Function expects columns {unknown_required} which are not in the dataset"
        )

    should_unpack_data: bool
    required_columns: set[str]
    if column_arguments:
        should_unpack_data = True
        required_columns = set(column_arguments)
        if "data" in parameter_names:
            raise ValueError(
                "data cannot be used as an argument to an evaluated function that explicitly takes columns as arguments"
            )
    elif "data" in parameter_names:
        # No column-named parameters: pass the full dataset under `data`.
        should_unpack_data = False
        required_columns = set(state.columns)
    else:
        raise ValueError(
            "Function must either explicitly take column names as arguments, or it must explicitly take a 'data' argument."
        )

    state = dsops.select(state, required_columns)
    if skip_evaluation_check:
        first_values = None
        eval_strategy = EvaluateStrategy(strategy)
    else:
        first_values, eval_strategy = do_first_evaluation(
            func,
            strategy,
            format,
            evaluator_kwargs,
            state,
            should_unpack_data,
        )
        if first_values is None and not allow_none:
            raise ValueError(
                "Cannot insert values from an evaluate function that returns None!"
            )

    if isinstance(first_values, dict):
        units = {
            name: val.unit if isinstance(val, Quantity) else None
            for name, val in first_values.items()
        }
        produces = set(first_values.keys())
    else:
        units = {
            func.__name__: first_values.unit
            if isinstance(first_values, Quantity)
            else None
        }
        produces = {func.__name__}

    column = EvaluatedColumn(
        func,
        required_columns,
        produces,
        format,
        units,
        eval_strategy,
        batch_size=batch_size,
        should_unpack_data=should_unpack_data,
        **evaluator_kwargs,
    )
    return column


def __verify(
    function: Callable, data_columns: Iterable[str], kwarg_names: Iterable[str]
):
    function_signature = signature(function)
    required_parameters = set()
    for name, parameter in function_signature.parameters.items():
        if parameter.default == Parameter.empty:
            required_parameters.add(name)

    missing = required_parameters.difference(data_columns).difference(kwarg_names)
    if not missing:
        return required_parameters.intersection(data_columns)
    elif len(missing) > 1:
        raise ValueError(
            f"All inputs to the function must either be column names or passed as keyword arguments! Found unknown input(s) {','.join(missing)}"
        )
