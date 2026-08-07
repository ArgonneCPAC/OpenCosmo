""" """

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from opencosmo.column.column import (
    Column,
    DerivedScalarValue,
    RawColumn,
)

if TYPE_CHECKING:
    from opencosmo import Dataset


class MissingColumnError(ValueError):
    pass


def get_column_selection(
    select_from: Iterable[str], select_by: Iterable[str]
) -> tuple[set[str], set[str]]:
    """
    Selects a list of columns from another list of columns. Supports wildcards.

    Returns two sets. The first is the set of matched columns. The second is the set
    of selections that could not be matched.
    """

    select_from = set(select_from)
    select_by = set(select_by)

    wildcards = set(n for n in select_by if "*" in n)
    complete = select_by - wildcards

    complete_matches = complete.intersection(select_from)
    complete_missing = complete.difference(select_from)

    if not any(wildcards):
        return complete_matches, complete_missing

    wildcard_matches = __evaluate_wildcards(select_from, wildcards)
    return complete_matches.union(wildcard_matches), complete_missing


def do_multi_dataset_selections(
    datasets: dict[str, Dataset],
    select_args: tuple[str | list[str], ...],
    select_kwargs: dict[str, Any],
    mode: str = "global",
):
    mode = select_kwargs.pop("mode", mode)
    columns_by_ds = {name: set(ds.columns) for name, ds in datasets.items()}
    length_by_ds = {name: len(ds) for name, ds in datasets.items()}
    args_by_ds, kwargs_by_ds = build_multi_dataset_selections(
        columns_by_ds, length_by_ds, select_args, select_kwargs
    )
    new_datasets = {}
    for name, dataset in datasets.items():
        ds_args = args_by_ds.get(name, [])
        ds_kwargs = kwargs_by_ds.get(name, {})
        if not ds_args and not ds_kwargs:
            new_datasets[name] = dataset
            continue
        try:
            new_ds = dataset.select(*ds_args, **ds_kwargs, mode=mode)
        # Do NOT fail if the only selections are wildcards, just return the raw dataset
        except MissingColumnError:
            if not all("*" in ds_arg for ds_arg in ds_args) or ds_kwargs:
                raise
            new_ds = dataset
        new_datasets[name] = new_ds
    return new_datasets


def build_multi_dataset_selections(
    columns_by_ds: dict[str, set[str]],
    ds_lengths: dict[str, int],
    select_args: tuple[str | list[str], ...],
    select_kwargs: dict[str, Any],
):
    flat_args = []
    for selection in select_args:
        if isinstance(selection, str):
            flat_args.append(selection)
        else:
            flat_args.extend(selection)
    assert set(columns_by_ds.keys()) == set(ds_lengths.keys())

    wildcards = []
    output_args: dict[Any, list[str]] = {name: [] for name in columns_by_ds.keys()}
    missing = set()
    for arg in flat_args:
        if not isinstance(arg, str):
            raise ValueError("Column selection arguments must be strings!")
        if "*" in arg:
            wildcards.append(arg)
            continue
        datasets_with_column = set(
            ds_name for ds_name, cols in columns_by_ds.items() if arg in cols
        )
        if not datasets_with_column:
            missing.add(arg)
            continue
        for ds_name in datasets_with_column:
            output_args[ds_name].append(arg)
    if missing:
        raise MissingColumnError(
            f"Columns {missing} could not be found in any datasets!"
        )
    for ds_selection in output_args.values():
        ds_selection.extend(wildcards)
    output_kwargs: dict[Any, dict[str, Any]] = {
        name: {} for name in columns_by_ds.keys()
    }
    missing = set()
    missing_lengths = set()
    for kwarg_name, kwarg_value in select_kwargs.items():
        if isinstance(kwarg_value, np.ndarray):
            col_length = len(kwarg_value)
            datasets_with_required_length = set(
                ds_name
                for ds_name, length in ds_lengths.items()
                if length == col_length
            )
            if not datasets_with_required_length:
                missing_lengths.add(kwarg_name)
                continue
            for ds_name in datasets_with_required_length:
                output_kwargs[ds_name][kwarg_name] = kwarg_value
            continue

        elif not isinstance(
            kwarg_value, (Column, DerivedScalarValue, RawColumn)
        ) and not hasattr(kwarg_value, "requires_names"):
            raise ValueError(
                f"Received unknown column type {kwarg_name} = {type(kwarg_value)}"
            )
        column_requires = kwarg_value.requires_names
        datasets_with_required_columns = set(
            ds_name
            for ds_name, ds_columns in columns_by_ds.items()
            if column_requires.issubset(ds_columns)
        )
        if not datasets_with_required_columns:
            missing.add(kwarg_name)
        for ds_name in datasets_with_required_columns:
            output_kwargs[ds_name][kwarg_name] = kwarg_value

    if missing:
        raise MissingColumnError(
            f"The columns required by {missing} do not exist in any of the datasets! Perhaps you mispelled some names?"
        )
    elif missing_lengths:
        raise ValueError(
            f"The arrays passed in {missing_lengths} do not match the length of any dataset!"
        )
    return output_args, output_kwargs


def __evaluate_wildcards(select_from: set[str], wildcards: Iterable[str]):
    wildcards = list(map(lambda w: w.replace("*", ".*"), wildcards))
    regex = re.compile("|".join(wildcards))
    matches = set(filter(lambda n: re.fullmatch(regex, n), select_from))
    return matches
