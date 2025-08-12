from typing import Any, Sequence

import numpy as np


def insert(
    storage: dict[str, np.ndarray], index: int, values_to_insert: dict[str, Any]
):
    if isinstance(values_to_insert, dict):
        for name, value in values_to_insert.items():
            storage[name][index] = value
        return storage

    name = next(iter(storage.keys()))
    storage[name][index] = values_to_insert


def prepare_kwargs(
    n_rows: int, evaluator_kwargs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Sequence]]:
    kwargs = {}
    array_kwargs = {}
    for name, kwarg in evaluator_kwargs.items():
        try:
            length = len(kwarg)
            if length == n_rows:
                array_kwargs[name] = kwarg
                continue
            kwargs[name] = kwarg

        except TypeError:
            kwargs[name] = kwarg

    return kwargs, array_kwargs
