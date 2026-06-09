from __future__ import annotations

from collections import defaultdict
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np
from astropy.units import Quantity

from opencosmo.column.column import EvaluatedColumn
from opencosmo.column.evaluate import EvaluateStrategy, do_first_evaluation
from opencosmo.dataset.formats import concat_chunks, fetch_as_dict

if TYPE_CHECKING:
    from opencosmo import Dataset

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
    dataset: Dataset,
    batch_size: int,
) -> dict[str, np.ndarray]:
    if column.batch_size > 0:
        return visit_dataset_batched(column, dataset)
    data = fetch_as_dict(dataset, column.requires_names, column.format, unpack=False)
    output = column.evaluate(data, dataset.index)
    if not isinstance(output, dict):
        assert len(column.produces) == 1
        output = {column.produces.pop(): output}
    return output


def visit_dataset_batched(column: EvaluatedColumn, dataset: Dataset):
    ranges = np.arange(0, len(dataset), column.batch_size)
    if ranges[-1] != len(dataset):
        ranges = np.append(ranges, len(dataset))

    output = defaultdict(list)

    for start, end in np.lib.stride_tricks.sliding_window_view(ranges, 2):
        batch_data = fetch_as_dict(
            dataset.take_range(start, end),
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
    dataset: Dataset,
    batch_size: int,
    allow_none=False,
    skip_evaluation_check=False,
) -> EvaluatedColumn:
    """
    Verify the function behaves correctly and determine the names of its output columns.
    """
    __verify(func, dataset.columns, evaluator_kwargs.keys())
    sig = signature(func)
    required_arguments = filter(
        lambda param: param.default == Parameter.empty, sig.parameters.values()
    )
    required_argument_names = set(map(lambda param: param.name, required_arguments))
    required_columns = required_argument_names.difference(evaluator_kwargs.keys())

    if diff := required_columns.difference(dataset.columns):
        raise ValueError(
            f"Function expects columns {diff} which are not in the dataset"
        )
    dataset = dataset.select(required_columns)
    if skip_evaluation_check:
        first_values = None
        eval_strategy = EvaluateStrategy(strategy)
    else:
        first_values, eval_strategy = do_first_evaluation(
            func, strategy, format, evaluator_kwargs, dataset
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
        batch_size,
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
