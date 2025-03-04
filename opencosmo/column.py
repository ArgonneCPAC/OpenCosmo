from __future__ import annotations

import operator as op
from typing import Callable

import astropy.units as u

Comparison = Callable[[float, float], bool]


def col(column_name: str) -> Column:
    return Column(column_name)


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
