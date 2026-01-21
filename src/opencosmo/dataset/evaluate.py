from __future__ import annotations

from inspect import Parameter, signature
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

from astropy.units import Quantity

from opencosmo.column.column import EvaluatedColumn
from opencosmo.column.evaluate import EvaluateStrategy, do_first_evaluation
from opencosmo.evaluate import (
    insert_data,
    make_output_from_first_values,
)

if TYPE_CHECKING:
    import numpy as np

    from opencosmo import Dataset

"""
Although the user-facing name for this operation is "evaluate", the pattern 
we are using here is known as a "visitor."
"""


def visit_dataset(
    function: Callable,
    strategy: str,
    format: str,
    evaluator_kwargs: dict[str, Any],
    dataset: Dataset,
):
    column = verify_for_lazy_evaluation(
        function,
        strategy,
        format,
        evaluator_kwargs,
        dataset,
        skip_evaluation_check=True,
    )

    data = dataset.select(column.requires).get_data(output=format)
    try:
        data = dict(data)
    except TypeError:
        data = {column.requires.pop(): data}
    output = column.evaluate(data, dataset.index)
    if not isinstance(output, dict):
        assert len(column.produces) == 1
        output = {column.produces.pop(): output}
    return output


def verify_for_lazy_evaluation(
    func: Callable,
    strategy: str,
    format: str,
    evaluator_kwargs: dict[str, Any],
    dataset: Dataset,
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
        produces = set(first_values.keys())
    else:
        produces = {func.__name__}
    column = EvaluatedColumn(
        func, required_columns, produces, format, eval_strategy, **evaluator_kwargs
    )
    return column


def __visit_rows_in_dataset(
    function: Callable,
    dataset: Dataset,
    format: str,
    kwargs: dict[str, Any] = {},
    iterable_kwargs: dict[str, Sequence] = {},
):
    first_row_values = dict(dataset.take(1, at="start").get_data())
    first_row_kwargs = kwargs | {name: arr[0] for name, arr in iterable_kwargs.items()}
    storage = __make_output(function, first_row_values | first_row_kwargs, len(dataset))
    for i, row in enumerate(dataset.rows(include_units=format == "astropy")):
        if i == 0:
            continue
        iter_kwargs = {name: arr[i] for name, arr in iterable_kwargs.items()}
        output = function(**row, **kwargs, **iter_kwargs)
        if storage is not None:
            insert_data(storage, i, output)
    return storage


def __visit_rows_in_data(
    function: Callable,
    data: dict[str, np.ndarray],
    format="astropy",
    kwargs: dict[str, Any] = {},
    iterable_kwargs: dict[str, np.ndarray] = {},
):
    data = {key: d for key, d in data.items() if key in signature(function).parameters}
    first_row_data = {name: arr[0] for name, arr in data.items()}
    first_row_kwargs = kwargs | {name: arr[0] for name, arr in iterable_kwargs.items()}
    n_rows = len(next(iter(data.values())))
    storage = __make_output(function, first_row_data | first_row_kwargs, n_rows)
    if format == "numpy":
        data = {
            key: arr.value if isinstance(arr, Quantity) else arr
            for key, arr in data.items()
        }

    for i in range(1, n_rows):
        row = {
            name: arr[i] for name, arr in chain(data.items(), iterable_kwargs.items())
        }
        output = function(**row, **kwargs)
        if storage is not None:
            insert_data(storage, i, output)
    return storage


def __make_output(
    function: Callable,
    first_input_values: dict[str, Any],
    n_rows: int,
) -> dict | None:
    first_values = function(**first_input_values)
    if first_values is None:
        return None
    if not isinstance(first_values, dict):
        name = function.__name__
        first_values = {name: first_values}

    return make_output_from_first_values(first_values, n_rows)


def __visit_vectorize(
    function: Callable,
    data: dict[str, Iterable] | Iterable,
    evaluator_kwargs: dict[str, Any] = {},
):
    pars = signature(function).parameters

    if not isinstance(data, dict) or (len(data) > 1 and len(pars) == 1):
        return function(data, **evaluator_kwargs)

    input_data = {pname: data[pname] for pname in pars if pname in data}

    return function(**input_data, **evaluator_kwargs)


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
