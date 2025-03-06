from __future__ import annotations
from functools import reduce

from typing import Callable
from astropy.table import Column # type: ignore

import typing
if typing.TYPE_CHECKING:
    from opencosmo.column.column import OcColumn

class DerivedColumn:
    """
    A column that is derived from other columns.
    """

    def __init__(
        self, columns: list[str], operations: list[Callable[[float, float], float]]
    ):
        self.columns = columns
        self.operations = operations

    def build(self, data: dict[str, Column]) -> Column:
        result = data[self.columns[0]] 
        for op, colname in zip(self.operations, self.columns[1:]):
            result = op(result, data[colname])
        return result

    def __mul__(self, other: "OcColumn") -> DerivedColumn: 
        return DerivedColumn(
            self.columns + [other.column_name],
            self.operations + [lambda x, y: x * y],
        )

    def __truediv__(self, other: "OcColumn") -> DerivedColumn:
        return DerivedColumn(
            self.columns + [other.column_name],
            self.operations + [lambda x, y: x / y],
        )
    
    def __add__(self, other: "OcColumn") -> DerivedColumn:
        return DerivedColumn(
            self.columns + [other.column_name],
            self.operations + [lambda x, y: x + y],
        )

    def __sub__(self, other: "OcColumn") -> DerivedColumn:
        return DerivedColumn(
            self.columns + [other.column_name],
            self.operations + [lambda x, y: x - y],
        )
