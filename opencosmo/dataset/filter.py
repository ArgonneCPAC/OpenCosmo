from __future__ import annotations

import operator as op
from collections import defaultdict
from typing import Callable

import astropy.units as u  # type: ignore
import numpy as np
from astropy import table  # type: ignore

from opencosmo.dataset.column import ColumnBuilder
from opencosmo.handler import OpenCosmoDataHandler

Comparison = Callable[[float, float], bool]


def col(column_name: str) -> Column:
    return Column(column_name)


def apply_filters(
    handler: OpenCosmoDataHandler,
    column_builders: dict[str, ColumnBuilder],
    filters: list[Filter],
    starting_filter: np.ndarray,
) -> np.ndarray:
    output_filter = starting_filter.copy()
    filters_by_column = defaultdict(list)
    for f in filters:
        filters_by_column[f.column_name].append(f)

    for column_name, column_filters in filters_by_column.items():
        column_filter = np.ones(output_filter.sum(), dtype=bool)
        builder = column_builders[column_name]
        column = handler.get_data({column_name: builder}, filter=output_filter)
        for f in column_filters:
            column_filter &= f.apply(column)
        output_filter[output_filter] &= column_filter
    return output_filter


class Column:
    """
    A column representa a column in the table. This is used first and foremost
    for filtering purposes. For example, if a user has loaded a dataset they
    can filter it with

    dataset.filter(oc.Col("column_name") < 5)

    In practice, this is just a factory class that returns filter
    """

    def __init__(self, column_name: str):
        self.column_name = column_name

    def __eq__(self, other: float | u.Quantity) -> Filter:
        return Filter(self.column_name, other, op.eq)

    def __ne__(self, other: float | u.Quantity) -> Filter:
        return Filter(self.column_name, other, op.ne)

    def __gt__(self, other: float | u.Quantity) -> Filter:
        return Filter(self.column_name, other, op.gt)

    def __ge__(self, other: float | u.Quantity) -> Filter:
        return Filter(self.column_name, other, op.ge)

    def __lt__(self, other: float | u.Quantity) -> Filter:
        return Filter(self.column_name, other, op.lt)

    def __le__(self, other: float | u.Quantity) -> Filter:
        return Filter(self.column_name, other, op.le)


class Filter:
    """
    A filter is a class that represents a filter on a column. It is used to
    filter a dataset.
    """

    def __init__(
        self, column_name: str, value: float | u.Quantity, operator: Comparison
    ):
        self.column_name = column_name
        self.value = value
        self.operator = operator

    def apply(self, column: table.Column) -> bool:
        """
        Filter the dataset based on the filter.
        """
        # Astropy's errors are good enough here
        if not isinstance(self.value, u.Quantity) and column.unit is not None:
            self.value *= column.unit
        return self.operator(column, self.value)
