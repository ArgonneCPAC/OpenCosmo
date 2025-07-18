from __future__ import annotations

import operator as op
from functools import partialmethod
from typing import Any, Callable, Iterable, Union

import astropy.units as u  # type: ignore
import numpy as np
from astropy import table  # type: ignore

Comparison = Callable[[float, float], bool]


class UnitsError(Exception):
    pass


def col(column_name: str) -> Column:
    """
    Create a reference to a column with a given name. These references can be combined
    to produce new columns or express queries that operate on the values in a given
    dataset. For example:

    .. code-block:: python

        import opencosmo as oc
        ds = oc.open("haloproperties.hdf5")
        query = oc.col("fof_halo_mass") > 1e14
        px = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
        ds = ds.with_new_columns(fof_halo_com_px = px).filter(query)

    For more advanced usage, see :doc:`cols`

    """
    return Column(column_name)


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
        match self.rhs:
            case Column():
                rhs_valid = self.rhs.column_name in names
            case DerivedColumn():
                rhs_valid = self.rhs.check_parent_existance(names)
            case _:
                rhs_valid = True

        match self.lhs:
            case Column():
                lhs_valid = self.lhs.column_name in names
            case DerivedColumn():
                lhs_valid = self.lhs.check_parent_existance(names)
            case _:
                lhs_valid = True

        return lhs_valid and rhs_valid

    def get_units(self, units: dict[str, u.Unit]):
        match self.lhs:
            case Column():
                lhs_unit = units[self.lhs.column_name]
            case DerivedColumn():
                lhs_unit = self.lhs.get_units(units)
            case _:
                lhs_unit = None
        match self.rhs:
            case Column():
                rhs_unit = units[self.rhs.column_name]
            case DerivedColumn():
                rhs_unit = self.rhs.get_units(units)
            case _:
                rhs_unit = None

        if self.operation in (op.sub, op.add):
            if lhs_unit != rhs_unit:
                raise UnitsError("Cannot add/subtract columns with different units!")
            return lhs_unit

        match (lhs_unit, rhs_unit):
            case (None, None):
                return None
            case (_, None):
                return self.operation(lhs_unit, 1)
            case (None, _):
                return self.operation(1, rhs_unit)
            case (_, _):
                return self.operation(lhs_unit, rhs_unit)

    def requires(self):
        """
        Return the raw data columns required to make this column
        """
        vals = set()
        match self.lhs:
            case Column():
                vals.add(self.lhs.column_name)
            case DerivedColumn():
                vals = vals | self.lhs.requires()
        match self.rhs:
            case Column():
                vals.add(self.rhs.column_name)
            case DerivedColumn():
                vals = vals | self.rhs.requires()

        return vals

    def combine_on_left(self, other: Column | DerivedColumn, operation: Callable):
        """
        Combine such that this column becomes the lhs of a new derived column.
        """
        match other:
            case Column() | DerivedColumn() | int() | float():
                return DerivedColumn(self, other, operation)
            case _:
                return NotImplemented

    def combine_on_right(self, other: Column | DerivedColumn, operation: Callable):
        """
        Combine such that this column becomes the rhs of a new derived column.
        """
        match other:
            case Column() | DerivedColumn() | int() | float():
                return DerivedColumn(other, self, operation)
            case _:
                return NotImplemented

    __mul__ = partialmethod(combine_on_left, operation=op.mul)
    __rmul__ = partialmethod(combine_on_right, operation=op.mul)
    __truediv__ = partialmethod(combine_on_left, operation=op.truediv)
    __rtruediv__ = partialmethod(combine_on_right, operation=op.truediv)
    __pow__ = partialmethod(combine_on_left, operation=op.pow)
    __add__ = partialmethod(combine_on_left, operation=op.add)
    __radd__ = partialmethod(combine_on_right, operation=op.add)
    __sub__ = partialmethod(combine_on_left, operation=op.sub)
    __rsub__ = partialmethod(combine_on_right, operation=op.sub)

    def evaluate(self, data: table.Table) -> table.Column:
        match self.rhs:
            case DerivedColumn():
                rhs = self.rhs.evaluate(data)
                rhs_data = rhs.value
                rhs_unit = rhs.unit
            case Column():
                rhs = data[self.rhs.column_name]
                rhs_data = rhs.value
                rhs_unit = rhs.unit
            case int() | float():
                rhs_data = self.rhs
                rhs_unit = None
        match self.lhs:
            case DerivedColumn():
                lhs = self.lhs.evaluate(data)
                lhs_data = lhs.value
                lhs_unit = lhs.unit
            case Column():
                lhs = data[self.lhs.column_name]
                lhs_data = lhs.value
                lhs_unit = lhs.unit
            case int() | float():
                lhs_data = self.lhs
                lhs_unit = None

        if self.operation in (op.add, op.sub):
            if lhs_unit != rhs_unit:
                raise ValueError("To add and subtract columns, units must be the same!")
            unit = lhs_unit
        elif self.operation == op.pow:
            if rhs_unit is not None:
                raise ValueError("Cannot raise values to powers with units!")
            if lhs_unit is not None:
                unit = self.operation(lhs_unit, rhs_data)
            else:
                unit = None

        else:
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
        values = self.operation(lhs_data, rhs_data)
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

    def __rmul__(self, other: Any) -> DerivedColumn:
        match other:
            case int() | float():
                return self * other
            case _:
                return NotImplemented

    def __mul__(self, other: Any) -> DerivedColumn:
        match other:
            case int() | float() | Column():
                return DerivedColumn(self, other, op.mul)
            case _:
                return NotImplemented

    def __rtruediv__(self, other: Any) -> DerivedColumn:
        match other:
            case int() | float():
                return DerivedColumn(other, self, op.truediv)
            case _:
                return NotImplemented

    def __truediv__(self, other: Any) -> DerivedColumn:
        match other:
            case int() | float() | Column():
                return DerivedColumn(self, other, op.truediv)
            case _:
                return NotImplemented

    def __pow__(self, other: Any) -> DerivedColumn:
        match other:
            case int() | float():
                return DerivedColumn(self, other, op.pow)
            case _:
                return NotImplemented

    def __add__(self, other: Any) -> DerivedColumn:
        match other:
            case Column():
                return DerivedColumn(self, other, op.add)
            case _:
                return NotImplemented

    def __sub__(self, other: Any) -> DerivedColumn:
        match other:
            case Column():
                return DerivedColumn(self, other, op.sub)
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

    def apply(self, column: table.Column | table.Table) -> np.ndarray:
        """
        mask the dataset based on the mask.
        """
        # Astropy's errors are good enough here
        if isinstance(column, table.Table):
            column = column[self.column_name]

        if isinstance(self.value, u.Quantity):
            if self.value.unit != column.unit:
                raise ValueError(
                    f"Incompatible units in fiter: {self.value.unit} and {column.unit}"
                )
            return self.operator(column.value, self.value.value)

        # mypy can't reason about columns correctly
        return self.operator(column.value, self.value)  # type: ignore
