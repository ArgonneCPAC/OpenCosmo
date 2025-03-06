from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

import astropy.units as u  # type: ignore
import numpy as np
from astropy import table  # type: ignore

from opencosmo.handler import handler
import typing

if typing.TYPE_CHECKING:
    from .column import ColumnBuilder

Comparison = Callable[[float, float], bool]


def apply_filters(
    handler: handler.OpenCosmoDataHandler,
    column_builders: dict[str, "ColumnBuilder"],
    filters: Iterable[Filter],
    starting_filter: np.ndarray,
) -> np.ndarray:
    output_filter = starting_filter.copy()
    filters_by_column = defaultdict(list)
    for f in filters:
        filters_by_column[f.column_name].append(f)

    column_names = set(column_builders.keys())
    filter_column_names = set(filters_by_column.keys())
    if not filter_column_names.issubset(column_names):
        raise ValueError(
            "Filters were applied to columns that do not exist in the dataset: "
            f"{filter_column_names - column_names}"
        )

    for column_name, column_filters in filters_by_column.items():
        column_filter = np.ones(output_filter.sum(), dtype=bool)
        builder = column_builders[column_name]
        column = handler.get_data({column_name: builder}, filter=output_filter)
        for f in column_filters:
            column_filter &= f.apply(column)
        output_filter[output_filter] &= column_filter
    return output_filter




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

    def apply(self, column: table.Column) -> np.ndarray:
        """
        Filter the dataset based on the filter.
        """
        # Astropy's errors are good enough here
        if not isinstance(self.value, u.Quantity) and column.unit is not None:
            self.value *= column.unit
        # mypy can't reason about columns correctly
        return self.operator(column, self.value)  # type: ignore
