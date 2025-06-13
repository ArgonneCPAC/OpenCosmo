from __future__ import annotations

import operator as op
from collections import defaultdict
from functools import partialmethod
from typing import TYPE_CHECKING, Any, Callable, Iterable, Union

import astropy.units as u  # type: ignore
import numpy as np
from astropy import table  # type: ignore

from opencosmo.dataset.column import ColumnBuilder
from opencosmo.index import DataIndex

if TYPE_CHECKING:
    from opencosmo.dataset.handler import DatasetHandler

Comparison = Callable[[float, float], bool]


def col(column_name: str) -> Column:
    return Column(column_name)


def apply_masks(
    handler: DatasetHandler,
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


ColumnOrScalar = Union["Column", "DerivedColumn", int, float]


class DerivedColumn:
    """
    A derived column represents a combination of multiple columns that already exist in
    the dataset through multiplication or division by other columns or scalars, which
    may or may not have units of their own.

    In general this is dangerous, because we cannot necessarily infer how a particular
    unit is supposed to respond to unit transformations. For the moment, we only allow
    for combinations of columns that already exist in the dataset.

    In general, columns that exist in the dataset are materialized first. Derived
    columns are then computed from these. The order of creation of the derived columns
    must be kept constant, in case you get another column which is derived from a
    derived column.
    """

    def __init__(self, lhs: ColumnOrScalar, rhs: ColumnOrScalar, operation: Callable):
        self.lhs = lhs
        self.rhs = rhs
        self.operation = operation

    def check_parent_existance(self, names: set[str]):
        if isinstance(self.rhs, Column):
            rhs_valid = self.rhs.column_name in names
        elif isinstance(self.rhs, DerivedColumn):
            rhs_valid = self.rhs.check_parent_existance(names)
        else:
            raise ValueError(f"Unknown type for rhs {type(self.rhs)}")

        if isinstance(self.lhs, Column):
            lhs_valid = self.lhs.column_name in names
        elif isinstance(self.lhs, DerivedColumn):
            lhs_valid = self.lhs.check_parent_existance(names)
        else:
            raise ValueError(f"Unknown type for rhs {type(self.rhs)}")

        return lhs_valid and rhs_valid

    def combine_on_left(self, other: Column | DerivedColumn, operation: Callable):
        """
        Combine such that this column becomes the lhs of a new derived column.
        """
        match other:
            case Column() | DerivedColumn():
                return DerivedColumn(self, other, operation)
            case _:
                raise NotImplementedError()

    def combine_on_right(self, other: Column | DerivedColumn, operation: Callable):
        """
        Combine such that this column becomes the rhs of a new derived column.
        """
        match other:
            case Column() | DerivedColumn():
                return DerivedColumn(other, self, operation)
            case _:
                raise NotImplementedError()

    __mul__ = partialmethod(combine_on_left, operation=op.mul)
    __rmul__ = partialmethod(combine_on_right, operation=op.mul)
    __truediv__ = partialmethod(combine_on_left, operation=op.truediv)
    __rtruediv__ = partialmethod(combine_on_right, operation=op.truediv)

    def evaluate(self, data: table.Table) -> table.Column:
        match self.rhs:
            case DerivedColumn():
                rhs = self.rhs.evaluate(data)
                rhs_unit = rhs.unit
            case Column():
                rhs = data[self.rhs.column_name]
                rhs_unit = rhs.unit
            case int() | float():
                rhs = self.rhs
                rhs_unit = None
        match self.lhs:
            case DerivedColumn():
                lhs = self.lhs.evaluate(data)
                lhs_unit = lhs.unit
            case Column():
                lhs = data[self.lhs.column_name]
                lhs_unit = lhs.unit
            case int() | float():
                lhs = self.rhs
                lhs_unit = None

        match (lhs_unit, rhs_unit):
            case (None, None):
                unit = None
            case (_, None):
                unit = lhs_unit
            case (None, _):
                unit = rhs_unit
            case _:
                unit = self.operation(lhs_unit, rhs_unit)

        # Astropy delegates __mul__ to the underlying numpy array, so we have
        # to manually handle units
        values = self.operation(lhs.data, rhs.data)
        if unit is not None:
            values *= unit
        return table.Column(values)


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
    def __eq__(self, other: float | u.Quantity) -> Mask:  # type: ignore
        return Mask(self.column_name, other, op.eq)

    def __ne__(self, other: float | u.Quantity) -> Mask:  # type: ignore
        return Mask(self.column_name, other, op.ne)

    def __gt__(self, other: float | u.Quantity) -> Mask:
        return Mask(self.column_name, other, op.gt)

    def __ge__(self, other: float | u.Quantity) -> Mask:
        return Mask(self.column_name, other, op.ge)

    def __lt__(self, other: float | u.Quantity) -> Mask:
        return Mask(self.column_name, other, op.lt)

    def __le__(self, other: float | u.Quantity) -> Mask:
        return Mask(self.column_name, other, op.le)

    def __mul__(self, other: Any) -> DerivedColumn:
        match other:
            case int() | float() | u.Quantity() | Column():
                return DerivedColumn(self, other, op.mul)
            case _:
                return NotImplemented

    def __truediv__(self, other: Any) -> DerivedColumn:
        match other:
            case int() | float() | u.Quantity() | Column():
                return DerivedColumn(self, other, op.truediv)
            case _:
                return NotImplemented

    def isin(self, other: Iterable[float | u.Quantity]) -> Mask:
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
