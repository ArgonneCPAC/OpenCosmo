from __future__ import annotations

import operator as op
from collections import defaultdict
from numbers import Real
from typing import Callable, Iterable

import astropy.units as u  # type: ignore
import numpy as np
from astropy import table  # type: ignore

from opencosmo.dataset.column import ColumnBuilder
from opencosmo.dataset.index import DataIndex
from opencosmo.handler import OpenCosmoDataHandler

Comparison = Callable[[float, float], bool]


def col(column_name: str) -> Column:
    return Column(column_name)


def apply_masks(
    handler: OpenCosmoDataHandler,
    column_builders: dict[str, ColumnBuilder],
    masks: Iterable[Mask],
    index: DataIndex,
) -> DataIndex:
    masks_by_column = defaultdict(list)
    for f in masks:
        masks_by_column[f.column_name].append(f)

    column_names = set(column_builders.keys())
    mask_column_names = set(masks_by_column.keys())
    if not mask_column_names.issubset(column_names):
        raise ValueError(
            "masks were applied to columns that do not exist in the dataset: "
            f"{mask_column_names - column_names}"
        )
    output_index = index

    for column_name, column_masks in masks_by_column.items():
        column_mask = np.ones(len(output_index), dtype=bool)
        builder = column_builders[column_name]
        column = handler.get_data({column_name: builder}, output_index)
        for f in column_masks:
            column_mask &= f.apply(column)
        output_index = output_index.mask(column_mask)
    return output_index


class Column:
    """
    A column representa a column in the table. This is used first and foremost
    for masking purposes. For example, if a user has loaded a dataset they
    can mask it with

    dataset.mask(oc.Col("column_name") < 5)

    In practice, this is just a factory class that returns mask
    """

    def __init__(self, column_name: str):
        self.column_name = column_name

    # mypy doesn't reason about eq and neq correctly
    def __eq__(self, other: Real | u.Quantity) -> Mask:  # type: ignore
        return Mask(self.column_name, other, op.eq)

    def __ne__(self, other: Real | u.Quantity) -> Mask:  # type: ignore
        return Mask(self.column_name, other, op.ne)

    def __gt__(self, other: Real | u.Quantity) -> Mask:
        return Mask(self.column_name, other, op.gt)

    def __ge__(self, other: Real | u.Quantity) -> Mask:
        return Mask(self.column_name, other, op.ge)

    def __lt__(self, other: Real | u.Quantity) -> Mask:
        return Mask(self.column_name, other, op.lt)

    def __le__(self, other: Real | u.Quantity) -> Mask:
        return Mask(self.column_name, other, op.le)

    def isin(self, other: Iterable[Real | u.Quantity]) -> Mask:
        return Mask(self.column_name, other, np.isin)


class Mask:
    """
    A mask is a class that represents a mask on a column. Masks evaluate
    to t/f for every element in the given column.
    """

    def __init__(
        self,
        column_name: str,
        value: float | u.Quantity,
        operator: Callable[[table.Column, float | u.Quantity], np.ndarray],
    ):
        self.column_name = column_name
        self.value = value
        self.operator = operator

    def apply(self, column: table.Column) -> np.ndarray:
        """
        mask the dataset based on the mask.
        """
        # Astropy's errors are good enough here
        value = self.value
        if not isinstance(value, u.Quantity) and column.unit is not None:
            value *= column.unit

        # mypy can't reason about columns correctly
        return self.operator(column, value)  # type: ignore
