from __future__ import annotations

from inspect import Parameter, signature
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

from astropy.table import Column, QTable  # type: ignore
from astropy.units import Quantity

from opencosmo.column.column import EvaluatedColumn, EvaluateStrategy
from opencosmo.column.evaluate import do_first_evaluation
from opencosmo.dataset.formats import convert_data
from opencosmo.evaluate import insert, make_output_from_first_values, prepare_kwargs

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
        function, strategy, format, evaluator_kwargs, dataset
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
