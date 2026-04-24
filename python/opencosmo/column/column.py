from __future__ import annotations

import operator as op
from copy import copy
from functools import partial, partialmethod
from inspect import signature
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
from astropy import table  # type: ignore

from opencosmo.column.evaluate import (
    EvaluateStrategy,
    evaluate_chunks,
    evaluate_rows,
    evaluate_vectorized,
)
from opencosmo.units import UnitsError

if TYPE_CHECKING:
    from uuid import UUID

    from opencosmo import Dataset
    from opencosmo.index import DataIndex

Comparison = Callable[[float, float], bool]
"""
The structures in this file are used both internally and user-facing.

The Column class represents a reference to a single column in the dataset. If
`is_raw` is set to true, we are requiring that this column is actually instantiated
from data originally in the hdf5 file. This is not intended to be set by the user,
and is only used internally.

A DerivedColumn represents a combination of columns that produces a single new column. 
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
    return Column(name)


ColumnOrScalar = Union["Column", "DerivedColumn", int, float]


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


class Column:
    """
    Represents a reference to a column with a given name. Column reference
    are created independently of the datasets that actually contain data.
    You should not create this class directly, instead use :py:meth:`opencosmo.col`.

    Columns can be combined, and support comparison operators for masking datasets.

    Combinations:

        - Basic arithmetic with +, -, \\*, and /
        - Powers with :code:`\\*\\*`, and :code:`column.sqrt()`
        - log and exponentiation with :code:`column.log10()` and :code:`column.exp10()`

    Comparison operators:

        - Arithmetic comparisons such as <, <=, >, ==, !=
        - Membership with :code:`column.isin`

    In general, combinations of columns produce a :code:`DerivedColumn`, which can be treated
    the exact same was as basic Columns.

    For example, to compute the x-component of a halo's momentum, and then filter out
    halos below a certain value of that momentum

    .. code-block:: python

        import opencosmo as oc

        dataset = oc.open("haloproperties.hdf5")
        halo_px = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
        dataset = dataset.with_new_columns(fof_halo_com_px = halo_px)

        min_momentum_filter = oc.col("fof_halo_com_px) > 10**14
        dataset = dataset.filter(min_momentum_filter)

    """

    def __init__(self, name: str):
        self.name = name
        self.description = None

    def __eq__(self, other: float | u.Quantity) -> ColumnMask:  # type: ignore
        return ColumnMask(self.name, other, op.eq)

    def __ne__(self, other: float | u.Quantity) -> ColumnMask:  # type: ignore
        return ColumnMask(self.name, other, op.ne)

    def __gt__(self, other: float | u.Quantity) -> ColumnMask:
        return ColumnMask(self.name, other, op.gt)

    def __ge__(self, other: float | u.Quantity) -> ColumnMask:
        return ColumnMask(self.name, other, op.ge)

    def __lt__(self, other: float | u.Quantity) -> ColumnMask:
        return ColumnMask(self.name, other, op.lt)

    def __le__(self, other: float | u.Quantity) -> ColumnMask:
        return ColumnMask(self.name, other, op.le)

    def isin(self, other: Iterable[float | u.Quantity]) -> ColumnMask:
        return ColumnMask(self.name, other, np.isin)

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

    def log10(self, unit_container: u.LogUnit = u.DexUnit) -> DerivedColumn:
        """
        Create a derived column that will compute the log of a given column. If
        the column contains units, the units must not be an astropy LogUnit
        (such as Dex or Mag)

        If you want the units of the new column to be a particular type of LogUnit,
        you can pass that type to the :code:`unit_container` argument. Defaults
        to DexUnit.
        """
        op = partial(_log10, unit_container=unit_container)
        return DerivedColumn(self, None, op)

    def exp10(self, expected_unit_container: u.LogUnit = u.DexUnit) -> DerivedColumn:
        """
        Create a derived column that will contain the base-10 exponentiation of the
        given column. If the column being exponentiated contains units, it must be an
        astropy LogUnit (e.g. Dex or Mag)

        You can specify the type of LogUnit container you expect the column to have with
        expected_unit_container. Defaults to DexUnit.
        """
        op = partial(_exp10, expected_unit_container=expected_unit_container)
        return DerivedColumn(self, None, op)

    def sqrt(self) -> DerivedColumn:
        """
        Create a derived column that will contain the square root of the given column.
        """
        return DerivedColumn(self, None, _sqrt)


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
    def __init__(self, name, description, alias=None, _dep_uuid=None):
        self.__name = name
        self.__description = description
        self.__alias = alias
        self.__uuid = uuid4()
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
            self.__name, self.__description, alias=self.__alias, _dep_uuid=dep_uuid
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

    def __init__(
        self,
        lhs: ColumnOrScalar,
        rhs: Optional[ColumnOrScalar],
        operation: Callable,
        description: Optional[str] = None,
        output_name: Optional[str] = None,
        _dep_map: dict[str, UUID] | None = None,
        no_cache: bool = False,
    ):
        self.lhs = lhs
        self.rhs = rhs
        self.name = output_name
        self.operation = operation
        self.description = description if description is not None else "None"
        self.__uuid = uuid4()
        self.__dep_map: dict[str, UUID] | None = _dep_map
        self.__no_cache = no_cache

    @property
    def uuid(self) -> UUID:
        return self.__uuid

    @property
    def dep_map(self) -> dict[str, UUID] | None:
        return self.__dep_map

    def bind(self, name_to_uuid: dict[str, UUID]) -> DerivedColumn:
        """
        Resolve each dependency column name to the UUID of the producer that was
        producing it at the time this column was registered with a dataset.
        Returns a new bound DerivedColumn; does not mutate this instance.
        """
        dep_map = {name: name_to_uuid[name] for name in self._traverse_names()}
        return DerivedColumn(
            self.lhs,
            self.rhs,
            self.operation,
            self.description,
            self.name,
            _dep_map=dep_map,
        )

    def _traverse_names(self) -> set[str]:
        """Walk the expression tree and collect all Column leaf names."""
        vals: set[str] = set()
        match self.lhs:
            case Column():
                vals.add(self.lhs.name)
            case DerivedColumn():
                vals |= self.lhs._traverse_names()
        match self.rhs:
            case Column():
                vals.add(self.rhs.name)
            case DerivedColumn():
                vals |= self.rhs._traverse_names()
        return vals

    @property
    def requires(self) -> set[UUID]:
        if self.__dep_map is None:
            raise RuntimeError(f"DerivedColumn '{self.name}' has not been bound yet.")
        return set(self.__dep_map.values())

    @property
    def produces(self):
        return None if self.name is None else set([self.name])

    @property
    def no_cache(self):
        return self.__no_cache

    def check_parent_existance(self, names: set[str]):
        match self.rhs:
            case Column():
                rhs_valid = self.rhs.name in names
            case DerivedColumn():
                rhs_valid = self.rhs.check_parent_existance(names)
            case _:
                rhs_valid = True

        match self.lhs:
            case Column():
                lhs_valid = self.lhs.name in names
            case DerivedColumn():
                lhs_valid = self.lhs.check_parent_existance(names)
            case _:
                lhs_valid = True

        return lhs_valid and rhs_valid

    def get_units(self, units: dict[str, u.Unit]):
        match self.lhs:
            case Column():
                lhs_unit = units[self.lhs.name]
            case DerivedColumn():
                lhs_unit = self.lhs.get_units(units)
            case _:
                lhs_unit = None
        match self.rhs:
            case Column():
                rhs_unit = units[self.rhs.name]
            case DerivedColumn():
                rhs_unit = self.rhs.get_units(units)
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

    def log10(self, unit_container=u.DexUnit):
        op = partial(_log10, unit_container=unit_container)
        return DerivedColumn(self, None, op)

    def exp10(self, expected_unit_container: u.LogUnit = u.DexUnit):
        op = partial(_exp10, expected_unit_container=expected_unit_container)
        return DerivedColumn(self, None, op)

    def sqrt(self):
        return DerivedColumn(self, None, _sqrt)

    def evaluate(self, data: dict[str, np.ndarray], *args) -> np.ndarray:
        lhs: np.typing.ArrayLike
        rhs: Optional[np.typing.ArrayLike]
        match self.lhs:
            case DerivedColumn():
                lhs = self.lhs.evaluate(data)
            case Column():
                lhs = data[self.lhs.name]
            case _:
                lhs = self.lhs
        match self.rhs:
            case DerivedColumn():
                rhs = self.rhs.evaluate(data)
            case Column():
                rhs = data[self.rhs.name]
            case _:
                rhs = self.rhs

        result = self.operation(lhs, rhs)
        return result


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
        self.__uuid = uuid4()
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
        if self.__format != "astropy":
            data = {
                name: val.value if isinstance(val, u.Quantity) else val
                for name, val in data.items()
            }

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
                return evaluate_rows(data, self.__func, self.__kwargs)
            case EvaluateStrategy.CHUNKED:
                if chunk_sizes is None:
                    raise ValueError(
                        "Cannot evaluate in CHUNKED strategy with a non-chunked index"
                    )
                return evaluate_chunks(data, self.__func, self.__kwargs, chunk_sizes)

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


class ColumnMask:
    """
    A mask is a class that represents a mask on a column. ColumnMasks evaluate
    to t/f for every element in the given column.
    """

    def __init__(
        self,
        name: str,
        value: float | u.Quantity,
        operator: Callable[[table.Column, float | u.Quantity], np.ndarray],
    ):
        self.name = name
        self.value = value
        self.operator = operator

    @property
    def requires(self):
        return {self.name}

    def apply(self, column: u.Quantity | np.ndarray) -> np.ndarray:
        """
        mask the dataset based on the mask.
        """
        # Astropy's errors are good enough here
        if isinstance(column, table.Table):
            column = column[self.name]

        if isinstance(self.value, u.Quantity) and isinstance(column, u.Quantity):
            if self.value.unit != column.unit:
                raise ValueError(
                    f"Incompatible units in fiter: {self.value.unit} and {column.unit}"
                )

        elif isinstance(column, u.Quantity):
            return self.operator(column.value, self.value)

        return self.operator(column, self.value)  # type: ignore

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

    def apply(self, data):
        left_mask = self.__left.apply(data)
        right_mask = self.__right.apply(data)
        return self.__op(left_mask, right_mask)
