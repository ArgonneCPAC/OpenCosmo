from __future__ import annotations

import operator as op
from copy import copy
from functools import partial, partialmethod, wraps
from inspect import currentframe, signature
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Optional,
    Protocol,
    Self,
    Union,
)
from uuid import uuid4

import astropy.units as u  # type: ignore
import numpy as np

from opencosmo.column.evaluate import (
    EvaluateStrategy,
    evaluate_chunks,
    evaluate_rows,
    evaluate_vectorized,
)
from opencosmo.units import UnitsError

if TYPE_CHECKING:
    from uuid import UUID

    from astropy import table

    from opencosmo import Dataset
    from opencosmo.index import DataIndex

Comparison = Callable[[float, float], bool]
"""
The structures in this file are used both internally and user-facing.

The Column class represents a reference to a single column in the dataset. If
`is_raw` is set to true, we are requiring that this column is actually instantiated
from data originally in the hdf5 file. This is not intended to be set by the user,
and is only used internally.

A Column represents a combination of columns that produces a single new column. 
The columns it depends on may or may not be raw columns themselves, but eventually
the dependency graph always points back to raw columns, or columns that were provided
directly by the user as numpy arrays/astropy quantities.

An evaluted column takes an arbitrary number of columns as input and returns an arbitrary
number of columns as output. These columns HAVE NOT actually been evaluated yet. 

These combinations all form a dependency graph, which can easily be evaluated for validity.

Raw columns and columns that are in-memory are allowed to be sources. All other columns
must take input. The dependency graph must be a DAG, where only those two types of columns
have no inputs.
"""


def ident(col, _):
    return col


def col(name: str) -> Column:
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
    return Column(name, None, ident)


def render_op(operation):
    if isinstance(operation, partial):
        operation = operation.func
    match operation:
        case op.mul:
            return "*"
        case op.truediv:
            return "/"
        case op.add:
            return "+"
        case op.sub:
            return "-"
        case op.pow:
            return "**"
        case f if f is _log10:
            return "log10"
        case f if f is _exp10:
            return "exp10"
        case f if f is _sqrt:
            return "sqrt"
        case f if f is _mean:
            return "mean"
        case f if f is _std:
            return "std"
        case f if f is _var:
            return "var"
        case f if f is _min:
            return "min"
        case f if f is _max:
            return "max"
        case f if f is _median:
            return "median"
        case f if f is _sum:
            return "sum"
        case f if f is _quantile:
            return "quantile"
        case f if f is _arcsin:
            return "arcsin"
        case f if f is _arccos:
            return "arccos"
        case f if f is _arctan2:
            return "arctan2"
        case _:
            return None


class Column:
    """
    A column represents a combination of one or more columns that already exist in
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

    def __init__(
        self,
        lhs: ColumnOrScalar,
        rhs: Optional[ColumnOrScalar],
        operation: Callable,
        description: Optional[str] = None,
        output_name: Optional[str] = None,
        _dep_map: dict[str, UUID] | None = None,
        no_cache: bool = False,
        _uuid: UUID | None = None,
    ):
        self.lhs = lhs
        self.rhs = rhs

        self.name = output_name
        self.operation = operation
        self.description = description if description is not None else "None"
        self.__uuid = _uuid if _uuid is not None else uuid4()
        self.__dep_map: dict[str, UUID] | None = _dep_map
        self.__no_cache = no_cache

    def __repr__(self):
        caller = currentframe().f_back.f_code.co_qualname
        if self.operation is ident:
            return self.lhs
        op_ = render_op(self.operation)
        if self.rhs is None:
            output = f"{op_}({self.lhs})"
        else:
            output = f"{self.lhs} {op_} {self.rhs}"

        if caller != "Column.__repr__":
            output = f"Column [ {output} ]"
        return output

    @property
    def uuid(self) -> UUID:
        return self.__uuid

    @property
    def dep_map(self) -> dict[str, UUID] | None:
        return self.__dep_map

    def bind(self, name_to_uuid: dict[str, UUID]) -> Column:
        """
        Resolve each dependency column name to the UUID of the producer that was
        producing it at the time this column was registered with a dataset.
        Returns a new bound Column; does not mutate this instance.
        """
        required_names = self._traverse_names()
        if missing := required_names.difference(name_to_uuid):
            raise ValueError(f"Derived column depends on unknown columns {missing}")

        dep_map = {name: name_to_uuid[name] for name in self._traverse_names()}
        return Column(
            self.lhs,
            self.rhs,
            self.operation,
            self.description,
            self.name,
            _dep_map=dep_map,
            _uuid=self.__uuid,
        )

    def with_global_scalars(self) -> Column:
        """
        Return a new Column with every nested DerivedScalarValue marked global.
        Does not mutate self.
        """
        new_lhs = _globalize(self.lhs)
        new_rhs = _globalize(self.rhs)
        return Column(
            new_lhs,
            new_rhs,
            self.operation,
            self.description,
            self.name,
            _dep_map=self.__dep_map,
            _uuid=self.__uuid,
        )

    def _traverse_names(self) -> set[str]:
        """Walk the expression tree and collect all Column leaf names."""
        vals: set[str] = set()
        match self.lhs:
            case str():
                vals.add(self.lhs)
            case Column() | DerivedScalarValue():
                vals |= self.lhs._traverse_names()
        match self.rhs:
            case str():
                vals.add(self.rhs)
            case Column() | DerivedScalarValue():
                vals |= self.rhs._traverse_names()
        return vals

    @property
    def requires(self) -> set[UUID]:
        if self.__dep_map is None:
            raise RuntimeError(f"Column '{self.name}' has not been bound yet.")
        return set(self.__dep_map.values())

    @property
    def produces(self):
        return None if self.name is None else set([self.name])

    @property
    def no_cache(self):
        return self.__no_cache

    def check_parent_existance(self, names: set[str]):
        match self.rhs:
            case str():
                rhs_valid = self.rhs in names
            case Column() | DerivedScalarValue():
                rhs_valid = self.rhs.check_parent_existance(names)
            case _:
                rhs_valid = True

        match self.lhs:
            case str():
                lhs_valid = self.lhs in names
            case Column() | DerivedScalarValue():
                lhs_valid = self.lhs.check_parent_existance(names)
            case _:
                lhs_valid = True

        return lhs_valid and rhs_valid

    def get_units(self, units: dict[str, u.Unit]):
        match self.lhs:
            case str():
                lhs_unit = units[self.lhs]
            case Column() | DerivedScalarValue():
                lhs_unit = self.lhs.get_units(units)
            case u.Quantity():
                lhs_unit = self.lhs.unit
            case _:
                lhs_unit = None
        match self.rhs:
            case str():
                rhs_unit = units[self.rhs]
            case Column() | DerivedScalarValue():
                rhs_unit = self.rhs.get_units(units)
            case u.Quantity():
                rhs_unit = self.rhs.unit
            case _:
                rhs_unit = None

        if self.operation in (op.sub, op.add) and (
            not isinstance(lhs_unit, u.LogUnit) or not isinstance(rhs_unit, u.LogUnit)
        ):
            if lhs_unit != rhs_unit:
                raise UnitsError("Cannot add/subtract columns with different units!")
            return lhs_unit

        match (lhs_unit, rhs_unit):
            case (None, None):
                return None
            case (_, None):
                if self.operation == op.pow:
                    return self.operation(lhs_unit, self.rhs)
                else:
                    return self.operation(lhs_unit, 1)
            case (None, _):
                return self.operation(1, rhs_unit)
            case (_, _):
                return self.operation(lhs_unit, rhs_unit)

    def combine_on_left(self, other: ColumnOrScalar, operation: Callable):
        """
        Combine such that this column becomes the lhs of a new derived column.
        """
        if isinstance(other, u.Quantity) and not other.isscalar:
            raise ValueError(
                f"Only scalar Quantity values can be used in column arithmetic, "
                f"got shape {other.shape}"
            )
        match other:
            case (
                str() | Column() | DerivedScalarValue() | int() | float() | u.Quantity()
            ):
                return Column(self, other, operation)
            case _:
                return NotImplemented

    def combine_on_right(self, other: ColumnOrScalar, operation: Callable):
        """
        Combine such that this column becomes the rhs of a new derived column.
        """
        if isinstance(other, u.Quantity) and not other.isscalar:
            raise ValueError(
                f"Only scalar Quantity values can be used in column arithmetic, "
                f"got shape {other.shape}"
            )
        match other:
            case (
                str() | Column() | DerivedScalarValue() | int() | float() | u.Quantity()
            ):
                return Column(other, self, operation)
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

    def log10(self, unit_container=u.DexUnit):
        op = partial(_log10, unit_container=unit_container)
        return Column(self, None, op)

    def exp10(self, expected_unit_container: u.LogUnit = u.DexUnit):
        op = partial(_exp10, expected_unit_container=expected_unit_container)
        return Column(self, None, op)

    def sqrt(self):
        return Column(self, None, _sqrt)

    def arcsin(self) -> Column:
        return Column(self, None, _arcsin)

    def arccos(self) -> Column:
        return Column(self, None, _arccos)

    def arctan2(self, other: ColumnOrScalar) -> Column:
        return Column(self, other, _arctan2)

    def mean(self) -> DerivedScalarValue:
        return DerivedScalarValue(self, None, _mean)

    def std(self) -> DerivedScalarValue:
        return DerivedScalarValue(self, None, _std)

    def var(self) -> DerivedScalarValue:
        return DerivedScalarValue(self, None, _var)

    def min(self) -> DerivedScalarValue:
        return DerivedScalarValue(self, None, _min)

    def max(self) -> DerivedScalarValue:
        return DerivedScalarValue(self, None, _max)

    def median(self) -> DerivedScalarValue:
        return DerivedScalarValue(self, None, _median)

    def sum(self) -> DerivedScalarValue:
        return DerivedScalarValue(self, None, _sum)

    def quantile(self, q: float) -> DerivedScalarValue:
        return DerivedScalarValue(self, None, partial(_quantile, q=q))

    def __eq__(self, other: float | u.Quantity) -> ColumnMask:  # type: ignore
        return ColumnMask(self, other, op.eq)

    def __ne__(self, other: float | u.Quantity) -> ColumnMask:  # type: ignore
        return ColumnMask(self, other, op.ne)

    def __gt__(self, other: float | u.Quantity) -> ColumnMask:
        return ColumnMask(self, other, op.gt)

    def __ge__(self, other: float | u.Quantity) -> ColumnMask:
        return ColumnMask(self, other, op.ge)

    def __lt__(self, other: float | u.Quantity) -> ColumnMask:
        return ColumnMask(self, other, op.lt)

    def __le__(self, other: float | u.Quantity) -> ColumnMask:
        return ColumnMask(self, other, op.le)

    def isin(self, other: Iterable[float | u.Quantity]) -> ColumnMask:
        return ColumnMask(self, other, np.isin)

    def evaluate(self, data: dict[str, np.ndarray], *args) -> np.ndarray:
        lhs: Any
        rhs: Any
        match self.lhs:
            case Column() | DerivedScalarValue():
                lhs = self.lhs.evaluate(data)
            case str():
                lhs = data[self.lhs]
            case _:
                lhs = self.lhs
        match self.rhs:
            case Column() | DerivedScalarValue():
                rhs = self.rhs.evaluate(data)
            case str():
                rhs = data[self.rhs]
            case _:
                rhs = self.rhs

        result = self.operation(lhs, rhs)
        return result


def _require_scalar_quantity(func: Callable) -> Callable:
    """Decorator that raises if any Quantity argument is non-scalar."""

    @wraps(func)
    def wrapper(self: Any, other: Any) -> Any:
        if isinstance(other, u.Quantity) and not other.isscalar:
            raise ValueError(
                f"Only scalar Quantity values can be used in column arithmetic, "
                f"got shape {other.shape}"
            )
        return func(self, other)

    return wrapper


ColumnOrScalar = Union[str, "Column", int, float, u.Quantity]


def _log10(
    left: np.ndarray | u.Unit,
    right: None,
    unit_container: u.LogUnit,
):
    vals = left
    unit = None
    if isinstance(left, u.UnitBase):
        return unit_container(left)

    elif isinstance(left, u.Quantity):
        vals = left.value
        unit = left.unit
        if isinstance(unit, u.LogUnit):
            raise ValueError("Cannot take the log of a log unit!")

    new_vals = np.log10(vals)
    if unit is not None:
        return new_vals * unit_container(unit)
    return new_vals


def _exp10(
    left: np.ndarray | u.Unit,
    right: None,
    expected_unit_container: u.LogUnit,
):
    vals = left
    unit = None
    if isinstance(left, u.LogUnit):
        if not isinstance(left, expected_unit_container):
            raise ValueError(
                f"Expected a unit of type {expected_unit_container}, found {type(left)}"
            )
        return left.physical_unit

    elif isinstance(left, u.Quantity):
        vals = left.value
        unit = left.unit
        if not isinstance(unit, u.LogUnit):
            raise ValueError(
                "Can only raise 10 to a unitful value if the unit is logarithmic"
            )
        if not isinstance(unit, expected_unit_container):
            raise ValueError(
                f"Expected a unit of type {expected_unit_container}, found {type(left)}"
            )

    new_vals = 10**vals
    if unit is not None:
        return new_vals * unit.physical_unit
    return new_vals


def _sqrt(left: np.ndarray | u.Unit, right: None):
    return left**0.5


def _mean(left: Any, right: None) -> Any:
    if isinstance(left, u.UnitBase):
        return left
    return np.mean(left)


def _std(left: Any, right: None) -> Any:
    if isinstance(left, u.UnitBase):
        return left
    return np.std(left)


def _var(left: Any, right: None) -> Any:
    if isinstance(left, u.UnitBase):
        return left**2
    return np.var(left)


def _min(left: Any, right: None) -> Any:
    if isinstance(left, u.UnitBase):
        return left
    return np.min(left)


def _max(left: Any, right: None) -> Any:
    if isinstance(left, u.UnitBase):
        return left
    return np.max(left)


def _median(left: Any, right: None) -> Any:
    if isinstance(left, u.UnitBase):
        return left
    return np.median(left)


def _sum(left: Any, right: None) -> Any:
    if isinstance(left, u.UnitBase):
        return left
    return np.sum(left)


def _quantile(left: Any, right: None, q: float) -> Any:
    if isinstance(left, u.UnitBase):
        return left
    return np.quantile(left, q)


def _require_dimensionless(unit: u.UnitBase, func_name: str) -> None:
    if not unit.is_equivalent(u.dimensionless_unscaled):
        raise UnitsError(
            f"{func_name} requires a dimensionless input, got unit '{unit}'"
        )


def _arcsin(left: Any, right: Any) -> Any:
    if isinstance(left, u.UnitBase):
        _require_dimensionless(left, "arcsin")
        return u.rad
    if isinstance(left, u.Quantity):
        _require_dimensionless(left.unit, "arcsin")
        return np.arcsin(left.value) * u.rad
    return np.arcsin(left)


def _arccos(left: Any, right: Any) -> Any:
    if isinstance(left, u.UnitBase):
        _require_dimensionless(left, "arccos")
        return u.rad
    if isinstance(left, u.Quantity):
        _require_dimensionless(left.unit, "arccos")
        return np.arccos(left.value) * u.rad
    return np.arccos(left)


def _arctan2(left: Any, right: Any) -> Any:
    left_is_unit = isinstance(left, u.UnitBase)
    right_is_unit = isinstance(right, u.UnitBase)
    if left_is_unit or right_is_unit:
        if not (left_is_unit and right_is_unit) or not left.is_equivalent(right):
            raise UnitsError(
                "arctan2 requires both inputs to have equivalent units or both to be unitless"
            )
        return u.rad
    left_is_qty = isinstance(left, u.Quantity)
    right_is_qty = isinstance(right, u.Quantity)
    if left_is_qty != right_is_qty:
        raise UnitsError(
            "arctan2 requires both inputs to have equivalent units or both to be unitless"
        )
    if left_is_qty:
        if not left.unit.is_equivalent(right.unit):
            raise UnitsError(
                f"arctan2 inputs have incompatible units: '{left.unit}' and '{right.unit}'"
            )
        return np.arctan2(left.value, right.value) * u.rad
    return np.arctan2(left, right)


class ConstructedColumn(Protocol):
    pass

    @property
    def uuid(self) -> UUID: ...

    @property
    def requires(self) -> set[UUID]: ...

    @property
    def dep_map(self) -> dict[str, UUID] | None: ...

    @property
    def produces(self) -> set[str]: ...

    @property
    def description(self) -> Optional[str]: ...

    def bind(self, name_to_uuid: dict[str, UUID]) -> Self: ...

    @property
    def no_cache(self) -> bool: ...
    def evaluate(
        self,
        data: dict[str, np.ndarray],
        index: DataIndex,
    ) -> np.ndarray | dict[str, np.ndarray]: ...

    def get_units(self, values: dict[str, u.Quantity]) -> dict[str, u.Unit]: ...


class RawColumn:
    def __init__(self, name, description, alias=None, _dep_uuid=None, _uuid=None):
        self.__name = name
        self.__description = description
        self.__alias = alias
        self.__uuid = _uuid if _uuid is not None else uuid4()
        self.__dep_uuid: UUID | None = _dep_uuid

    @property
    def uuid(self) -> UUID:
        return self.__uuid

    @property
    def name(self):
        return self.__name

    def bind(self, name_to_uuid: dict[str, UUID]) -> RawColumn:
        if self.__alias is None:
            return self
        dep_uuid = name_to_uuid[self.__name]
        return RawColumn(
            self.__name,
            self.__description,
            alias=self.__alias,
            _dep_uuid=dep_uuid,
            _uuid=self.__uuid,
        )

    @property
    def requires(self) -> set[UUID]:
        if self.__alias is None:
            return set()
        if self.__dep_uuid is None:
            raise RuntimeError(
                f"RawColumn alias '{self.__alias}' has not been bound yet."
            )
        return {self.__dep_uuid}

    @property
    def dep_map(self) -> dict[str, UUID]:
        if self.__alias is None:
            return {}
        if self.__dep_uuid is None:
            raise RuntimeError(
                f"RawColumn alias '{self.__alias}' has not been bound yet."
            )
        return {self.__name: self.__dep_uuid}

    @property
    def no_cache(self):
        return False

    @property
    def alias(self) -> str | None:
        return self.__alias

    @property
    def produces(self) -> set[str]:
        return set([self.__alias or self.__name])

    @property
    def description(self):
        return self.__description

    def get_units(self, values: dict[str, u.Quantity]) -> dict[str, u.Unit]:
        return values[self.__name]

    def evaluate(self, data: dict[str, np.ndarray], *args):
        return data[self.__name]


class DerivedScalarValue:
    """
    A scalar value derived from a column reduction (mean, std, quantile, ...) or
    from arithmetic between other scalar values. Used inside column arithmetic to
    express things like column normalization: :code:`(col - col.mean()) / col.std()`.

    Reductions are computed over the rows that are materialized at evaluation time,
    so applying a filter or :code:`bound` before the reduction changes the result.
    """

    def __init__(
        self,
        lhs: ColumnOrScalar | DerivedScalarValue,
        rhs: ColumnOrScalar | DerivedScalarValue | None,
        operation: Callable,
        description: Optional[str] = None,
        output_name: Optional[str] = None,
        _dep_map: dict[str, UUID] | None = None,
        no_cache: bool = True,
        _uuid: UUID | None = None,
        is_global: bool = False,
    ):
        self.lhs = lhs
        self.rhs = rhs
        self.operation = operation
        self.name = output_name
        self.description = description if description is not None else "None"
        self.__uuid = _uuid if _uuid is not None else uuid4()
        self.__dep_map: dict[str, UUID] | None = _dep_map
        self.__no_cache = no_cache
        self.__is_global = is_global

    def _traverse_names(self) -> set[str]:
        vals: set[str] = set()
        if isinstance(self.lhs, (DerivedScalarValue, Column)):
            vals |= self.lhs._traverse_names()
        if isinstance(self.rhs, (DerivedScalarValue, Column)):
            vals |= self.rhs._traverse_names()
        return vals

    @property
    def uuid(self) -> UUID:
        return self.__uuid

    @property
    def dep_map(self) -> dict[str, UUID] | None:
        return self.__dep_map

    @property
    def requires(self) -> set[UUID]:
        if self.__dep_map is None:
            raise RuntimeError(
                f"DerivedScalarValue '{self.name}' has not been bound yet."
            )
        return set(self.__dep_map.values())

    @property
    def produces(self):
        return None if self.name is None else set([self.name])

    @property
    def no_cache(self):
        return self.__no_cache

    @property
    def is_global(self) -> bool:
        return self.__is_global

    def bind(self, name_to_uuid: dict[str, UUID]) -> DerivedScalarValue:
        """
        Resolve each dependency column name to the UUID of the producer that was
        producing it at the time this scalar was registered with a dataset.
        Returns a new bound DerivedScalarValue; does not mutate this instance.
        """
        required_names = self._traverse_names()
        if missing := required_names.difference(name_to_uuid):
            raise ValueError(f"Derived scalar depends on unknown columns {missing}")

        dep_map = {name: name_to_uuid[name] for name in self._traverse_names()}
        return DerivedScalarValue(
            self.lhs,
            self.rhs,
            self.operation,
            self.description,
            self.name,
            _dep_map=dep_map,
            no_cache=self.__no_cache,
            _uuid=self.__uuid,
            is_global=self.__is_global,
        )

    def with_global(self) -> DerivedScalarValue:
        """
        Return a new DerivedScalarValue marked as global, with the flag recursively
        set on any nested DerivedScalarValue in lhs/rhs (including inside Column
        subtrees). Does not mutate self.
        """
        return DerivedScalarValue(
            _globalize(self.lhs),
            _globalize(self.rhs),
            self.operation,
            description=self.description,
            output_name=self.name,
            _dep_map=self.__dep_map,
            no_cache=self.__no_cache,
            _uuid=self.__uuid,
            is_global=True,
        )

    def __repr__(self):
        op_str = render_op(self.operation)
        if self.rhs is None:
            return f"{op_str}({self.lhs})"
        return f"{self.lhs} {op_str} {self.rhs}"

    def check_parent_existance(self, names: set[str]) -> bool:
        match self.lhs:
            case Column() | DerivedScalarValue():
                lhs_valid = self.lhs.check_parent_existance(names)
            case _:
                lhs_valid = True
        match self.rhs:
            case Column() | DerivedScalarValue():
                rhs_valid = self.rhs.check_parent_existance(names)
            case _:
                rhs_valid = True
        return lhs_valid and rhs_valid

    def get_units(self, units: dict[str, u.Unit]):
        match self.lhs:
            case Column() | DerivedScalarValue():
                lhs_unit = self.lhs.get_units(units)
            case u.Quantity():
                lhs_unit = self.lhs.unit
            case _:
                lhs_unit = None
        match self.rhs:
            case Column() | DerivedScalarValue():
                rhs_unit = self.rhs.get_units(units)
            case u.Quantity():
                rhs_unit = self.rhs.unit
            case _:
                rhs_unit = None

        if self.operation in (op.sub, op.add):
            if lhs_unit != rhs_unit:
                raise UnitsError("Cannot add/subtract scalars with different units!")
            return lhs_unit

        match (lhs_unit, rhs_unit):
            case (None, None):
                return None
            case (_, None):
                if self.operation == op.pow:
                    return self.operation(lhs_unit, self.rhs)
                return self.operation(lhs_unit, 1)
            case (None, _):
                return self.operation(1, rhs_unit)
            case (_, _):
                return self.operation(lhs_unit, rhs_unit)

    def evaluate(self, data: dict[str, np.ndarray], *args) -> Any:
        match self.lhs:
            case Column() | DerivedScalarValue():
                lhs = self.lhs.evaluate(data)
            case _:
                lhs = self.lhs
        match self.rhs:
            case Column() | DerivedScalarValue():
                rhs = self.rhs.evaluate(data)
            case _:
                rhs = self.rhs
        if self.__is_global and rhs is None:
            from opencosmo.column.reductions_mpi import evaluate_global_reduction

            return evaluate_global_reduction(self.operation, lhs)
        return self.operation(lhs, rhs)

    def _combine_on_left(self, other: Any, operation: Callable):
        if isinstance(other, u.Quantity) and not other.isscalar:
            raise ValueError(
                f"Only scalar Quantity values can be used in column arithmetic, "
                f"got shape {other.shape}"
            )
        match other:
            case DerivedScalarValue() | int() | float() | u.Quantity():
                return DerivedScalarValue(
                    self, other, operation, description=None, output_name=None
                )
            case Column():
                return Column(self, other, operation)
            case _:
                return NotImplemented

    def _combine_on_right(self, other: Any, operation: Callable):
        if isinstance(other, u.Quantity) and not other.isscalar:
            raise ValueError(
                f"Only scalar Quantity values can be used in column arithmetic, "
                f"got shape {other.shape}"
            )
        match other:
            case DerivedScalarValue() | int() | float() | u.Quantity():
                return DerivedScalarValue(
                    other, self, operation, description=None, output_name=None
                )
            case Column():
                return Column(other, self, operation)
            case _:
                return NotImplemented

    __mul__ = partialmethod(_combine_on_left, operation=op.mul)
    __rmul__ = partialmethod(_combine_on_right, operation=op.mul)
    __truediv__ = partialmethod(_combine_on_left, operation=op.truediv)
    __rtruediv__ = partialmethod(_combine_on_right, operation=op.truediv)
    __add__ = partialmethod(_combine_on_left, operation=op.add)
    __radd__ = partialmethod(_combine_on_right, operation=op.add)
    __sub__ = partialmethod(_combine_on_left, operation=op.sub)
    __rsub__ = partialmethod(_combine_on_right, operation=op.sub)

    def __pow__(self, other: Any):
        match other:
            case int() | float():
                return DerivedScalarValue(self, other, op.pow)
            case _:
                return NotImplemented

    def __eq__(self, other: float | u.Quantity) -> ColumnMask:  # type: ignore
        return ColumnMask(self, other, op.eq)

    def __ne__(self, other: float | u.Quantity) -> ColumnMask:  # type: ignore
        return ColumnMask(self, other, op.ne)

    def __gt__(self, other: float | u.Quantity) -> ColumnMask:
        return ColumnMask(self, other, op.gt)

    def __ge__(self, other: float | u.Quantity) -> ColumnMask:
        return ColumnMask(self, other, op.ge)

    def __lt__(self, other: float | u.Quantity) -> ColumnMask:
        return ColumnMask(self, other, op.lt)

    def __le__(self, other: float | u.Quantity) -> ColumnMask:
        return ColumnMask(self, other, op.le)


class EvaluatedColumn:
    def __init__(
        self,
        func: Callable,
        requires: set[str],
        produces: set[str],
        format: str,
        units: dict[str, Optional[u.Unit]],
        strategy: EvaluateStrategy = EvaluateStrategy.ROW_WISE,
        batch_size: int = -1,
        description: Optional[str] = None,
        _dep_map: dict[str, UUID] | None = None,
        no_cache: bool = False,
        _uuid: UUID | None = None,
        **kwargs: Any,
    ):
        self.__func = func
        self.__requires = requires
        self.__kwargs = kwargs
        self.__produces = produces
        self.__units = units
        self.__format = format
        self.__strategy = strategy
        self.__batch_size = batch_size
        self.__no_cache = no_cache
        self.description = description
        self.__uuid = _uuid if _uuid is not None else uuid4()
        self.__dep_map = _dep_map

    @property
    def uuid(self) -> UUID:
        return self.__uuid

    @property
    def dep_map(self) -> dict[str, UUID] | None:
        return self.__dep_map

    def bind(self, name_to_uuid: dict[str, UUID]) -> EvaluatedColumn:
        """
        Resolve each dependency column name to the UUID of the producer that was
        producing it at the time this column was registered with a dataset.
        Returns a new bound EvaluatedColumn; does not mutate this instance.
        """
        dep_map = {name: name_to_uuid[name] for name in self.__requires}
        return EvaluatedColumn(
            self.__func,
            self.__requires,
            self.__produces,
            self.__format,
            self.__units,
            self.__strategy,
            self.__batch_size,
            self.description,
            _dep_map=dep_map,
            no_cache=self.__no_cache,
            _uuid=self.__uuid,
            **self.__kwargs,
        )

    @property
    def requires_names(self) -> set[str]:
        """Return the required column names (as strings), for use in data lookup."""
        if self.__dep_map is not None:
            return set(self.__dep_map.keys())
        return copy(self.__requires)

    def with_kwargs(self, **new_kwargs: Any):
        new_kwargs = self.__kwargs | new_kwargs
        return EvaluatedColumn(
            self.__func,
            self.__requires,
            self.__produces,
            self.__format,
            self.__units,
            self.__strategy,
            self.__batch_size,
            self.description,
            _dep_map=self.__dep_map,
            **new_kwargs,
        )

    @property
    def name(self):
        return self.__func.__name__

    @property
    def requires(self) -> set[UUID]:
        if self.__dep_map is None:
            raise RuntimeError(f"EvaluatedColumn '{self.name}' has not been bound yet.")
        return set(self.__dep_map.values())

    @property
    def no_cache(self):
        return self.__no_cache

    @property
    def produces(self):
        return copy(self.__produces)

    @property
    def signature(self):
        return signature(self.__func)

    @property
    def strategy(self):
        return self.__strategy

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def format(self):
        return self.__format

    @property
    def kwarg_names(self):
        return self.__kwargs.keys()

    def get_units(self, units: dict[str, np.ndarray]):
        return self.__units

    def evaluate(self, data: dict[str, np.ndarray], index: DataIndex | None):
        data = {name: data[name] for name in self.__requires}
        chunk_sizes = index[1] if isinstance(index, tuple) else None

        if self.batch_size > 0:
            length = len(next(iter(data.values())))
            strategy = EvaluateStrategy.CHUNKED
            chunk_sizes = np.full(
                np.ceil(length / self.batch_size).astype(int), self.batch_size
            )
            chunk_sizes[-1] = (length % self.batch_size) or self.batch_size

        else:
            strategy = self.__strategy

        match strategy:
            case EvaluateStrategy.VECTORIZE:
                return evaluate_vectorized(data, self.__func, self.__kwargs, index)
            case EvaluateStrategy.ROW_WISE:
                return evaluate_rows(data, self.__func, self.__kwargs, self.__format)
            case EvaluateStrategy.CHUNKED:
                if chunk_sizes is None:
                    raise ValueError(
                        "Cannot evaluate in CHUNKED strategy with a non-chunked index"
                    )
                return evaluate_chunks(
                    data, self.__func, self.__kwargs, chunk_sizes, self.__format
                )

    def evaluate_for_storage(
        self, data: dict[str, np.ndarray], index: DataIndex | None
    ) -> dict[str, np.ndarray]:
        """
        Evaluate and return numpy-formatted output suitable for the column
        cache. Input arrives in the numpy/astropy form used internally, so
        it is first converted to the user's requested format before the
        function runs, and the output is converted back to numpy.
        """
        from opencosmo.dataset.formats import to_format_dict, to_numpy_dict

        required = {name: data[name] for name in self.__requires}
        converted = to_format_dict(required, self.__format)
        output = self.evaluate(converted, index)
        if not isinstance(output, dict):
            output = {next(iter(self.__produces)): output}
        return to_numpy_dict(output)

    def evaluate_one(self, dataset: Dataset):
        match self.__strategy:
            case EvaluateStrategy.VECTORIZE:
                values = (
                    dataset.select(self.__requires)
                    .take(1)
                    .get_data(self.__format, unpack=False)
                )
                values = dict(values)
                return self.__func(**values, **self.__kwargs)

            case EvaluateStrategy.ROW_WISE:
                values = (
                    dataset.select(self.__requires)
                    .take(1)
                    .get_data(self.__format, unpack=True)
                )
                values = dict(values)
                return self.__func(**values, **self.__kwargs)

            case EvaluateStrategy.CHUNKED:
                index = dataset.index
                assert isinstance(index, tuple)
                first_chunk_size = index[1][0]
                first_chunk = (
                    dataset.select(self.__requires)
                    .take(first_chunk_size)
                    .get_data(self.__format)
                )
                first_chunk = dict(first_chunk)
                return self.__func(**first_chunk, **self.__kwargs)

        pass


def _evaluate_scalar(scalar: DerivedScalarValue, ds: Dataset) -> Any:
    """
    Materialize the columns a DerivedScalarValue depends on and evaluate
    the reduction against them. Pulls data via the astropy format so Quantity
    units survive into the reduction.
    """
    required = scalar._traverse_names()
    if not required:
        return scalar.evaluate({})
    table = ds.select(*required).get_data("astropy", unpack=False, wrap_single=True)
    data = {name: table[name] for name in required}
    return scalar.evaluate(data)


class ColumnMask:
    """
    A mask is a class that represents a mask on a column. ColumnMasks evaluate
    to t/f for every element in the given column.
    """

    def __init__(
        self,
        left: ColumnOrScalar,
        right: ColumnOrScalar,
        operator: Callable[[table.Column, float | u.Quantity], np.ndarray],
    ):
        self.left = left
        self.right = right
        self.operator = operator

    def apply(self, ds: Dataset):
        match self.left:
            case Column():
                left = ds.select(data=self.left).get_data()
            case DerivedScalarValue():
                left = _evaluate_scalar(self.left, ds)
            case _:
                left = self.left

        right_selected = False
        match self.right:
            case Column():
                right = ds.select(data=self.right).get_data()
                right_selected = True
            case DerivedScalarValue():
                right = _evaluate_scalar(self.right, ds)
            case _:
                right = self.right
        if (
            isinstance(left, u.Quantity)
            and not isinstance(right, u.Quantity)
            and not right_selected
        ):
            return self.operator(left.value, right)
        result = self.operator(left, right)
        return result

    def __and__(self, other: Self | CompoundColumnMask):
        return CompoundColumnMask(self, other, lambda left, right: left & right)

    def __or__(self, other: Self | CompoundColumnMask):
        return CompoundColumnMask(self, other, lambda left, right: left | right)


class CompoundColumnMask:
    def __init__(
        self,
        left: ColumnMask | Self,
        right: ColumnMask | Self,
        op: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        self.__left = left
        self.__right = right
        self.__op = op

    @property
    def requires(self):
        columns = set()
        columns |= self.__left.requires
        columns |= self.__right.requires
        return columns

    def __and__(self, other: ColumnMask | Self):
        return CompoundColumnMask(self, other, lambda left, right: left & right)

    def __or__(self, other: ColumnMask | Self):
        return CompoundColumnMask(self, other, lambda left, right: left | right)

    def apply(self, ds: Dataset):
        left_mask = self.__left.apply(ds)
        right_mask = self.__right.apply(ds)
        return self.__op(left_mask, right_mask)


def _globalize(node: Any) -> Any:
    """
    Helper to recursively mark DerivedScalarValue nodes as global within a tree.
    Used by Column.with_global_scalars() and DerivedScalarValue.with_global().
    """
    if isinstance(node, DerivedScalarValue):
        return node.with_global()
    if isinstance(node, Column):
        return node.with_global_scalars()
    return node
