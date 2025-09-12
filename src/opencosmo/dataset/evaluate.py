from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

from astropy.table import Column, QTable  # type: ignore

from opencosmo.evaluate import insert, make_output_from_first_values, prepare_kwargs

if TYPE_CHECKING:
    from opencosmo import Dataset

"""
Although the user-facing name for this operation is "evaluate", the pattern 
we are using here is known as a "visitor."
"""


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
        if result is not None and not isinstance(result, dict):
            return {function.__name__: result}
        return result
    else:
        kwargs, iterable_kwargs = prepare_kwargs(len(dataset), evaluator_kwargs)
        return __visit_rows(function, dataset, format, kwargs, iterable_kwargs)


def __visit_rows(
    function: Callable,
    dataset: "Dataset",
    format="astropy",
    kwargs: dict[str, Any] = {},
    iterable_kwargs: dict[str, Sequence] = {},
):
    requested_columns = (
        set(signature(function).parameters)
        - set(kwargs.keys())
        - set(iterable_kwargs.keys())
    )
    using_all_columns = len(dataset.columns) > 1 and len(requested_columns) == 1

    first_row_kwargs = kwargs | {name: arr[0] for name, arr in iterable_kwargs.items()}
    storage = __make_output(function, dataset, using_all_columns, first_row_kwargs)
    for i, row in enumerate(dataset.rows(output=format)):
        if i == 0:
            continue
        iter_kwargs = {name: arr[i] for name, arr in iterable_kwargs.items()}
        if using_all_columns:
            output = function(row, **kwargs, **iter_kwargs)
        else:
            output = function(**row, **kwargs, **iter_kwargs)
        if storage is not None:
            insert(storage, i, output)
    return storage


def __make_output(
    function: Callable,
    dataset: "Dataset",
    using_all_columns: bool,
    kwargs: dict[str, Any],
) -> dict | None:
    if using_all_columns:
        first_values = function(next(dataset.take(1, at="start").rows()), **kwargs)
    else:
        first_values = function(**next(dataset.take(1, at="start").rows()), **kwargs)
    n_rows = len(dataset)
    if first_values is None:
        return None
    if not isinstance(first_values, dict):
        name = function.__name__
        first_values = {name: first_values}

    return make_output_from_first_values(first_values, n_rows)


def __visit_vectorize(
    function: Callable,
    dataset: "Dataset",
    format: str = "astropy",
    evaluator_kwargs: dict[str, Any] = {},
):
    data = dataset.get_data(format)
    if format == "astropy" and isinstance(data, QTable):
        data = {name: col for name, col in data.items()}
    elif isinstance(data, Column):
        data = {data.name: data.quantity}

    if not isinstance(data, dict) or (
        len(data) > 1 and len(signature(function).parameters) == 1
    ):
        return function(data, **evaluator_kwargs)
    return function(**data, **evaluator_kwargs)


def __prepare(function: Callable, dataset: "Dataset", evaluator_kwargs: Iterable[str]):
    function_arguments = set(signature(function).parameters.keys())

    input_columns = function_arguments.intersection(dataset.columns)
    if len(input_columns) == 0 and dataset.dtype in function_arguments:
        return dataset
    return dataset.select(input_columns)


def __verify(function: Callable, dataset: "Dataset", kwarg_names: Iterable[str]):
    function_signature = signature(function)
    required_parameters = set()
    for name, parameter in function_signature.parameters.items():
        if parameter.default == Parameter.empty:
            required_parameters.add(name)

    dataset_columns = set(dataset.columns)
    kwarg_names = set(kwarg_names)
    missing = required_parameters - dataset_columns - kwarg_names
    if not missing:
        return
    elif len(missing) > 1 or next(iter(missing)) != dataset.dtype:
        raise ValueError(
            f"All inputs to the function must either be column names or passed as keyword arguments! Found unknown input(s) {','.join(missing)}"
        )
