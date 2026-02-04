from __future__ import annotations

import operator as op
from copy import copy
from functools import cached_property, partial, partialmethod
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
    from opencosmo import Dataset
    from opencosmo.index import DataIndex

Comparison = Callable[[float, float], bool]


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

    def __init__(self, column_name: str):
        self.column_name = column_name

    # mypy doesn't reason about eq and neq correctly
    def __eq__(self, other: float | u.Quantity) -> ColumnMask:  # type: ignore
        return ColumnMask(self.column_name, other, op.eq)

    def __ne__(self, other: float | u.Quantity) -> ColumnMask:  # type: ignore
        return ColumnMask(self.column_name, other, op.ne)

    def __gt__(self, other: float | u.Quantity) -> ColumnMask:
        return ColumnMask(self.column_name, other, op.gt)

    def __ge__(self, other: float | u.Quantity) -> ColumnMask:
        return ColumnMask(self.column_name, other, op.ge)

    def __lt__(self, other: float | u.Quantity) -> ColumnMask:
        return ColumnMask(self.column_name, other, op.lt)

    def __le__(self, other: float | u.Quantity) -> ColumnMask:
        return ColumnMask(self.column_name, other, op.le)

    def isin(self, other: Iterable[float | u.Quantity]) -> ColumnMask:
        return ColumnMask(self.column_name, other, np.isin)

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
    def requires(self) -> set[str]: ...
    @property
    def produces(self) -> Optional[set[str]]: ...
    @property
    def description(self) -> Optional[str]: ...

    def evaluate(
        self,
        data: dict[str, np.ndarray],
        index: DataIndex,
    ) -> np.ndarray | dict[str, np.ndarray]: ...

    def get_units(self, values: dict[str, u.Quantity]) -> dict[str, u.Unit]: ...


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
    ):
        self.lhs = lhs
        self.rhs = rhs
        self.operation = operation
        self.description = description if description is not None else "None"

    @cached_property
    def requires(self):
        """
        Return the raw data columns required to make this column
        """
        vals = set()
        match self.lhs:
            case Column():
                vals.add(self.lhs.column_name)
            case DerivedColumn():
                vals = vals | self.lhs.requires
        match self.rhs:
            case Column():
                vals.add(self.rhs.column_name)
            case DerivedColumn():
                vals = vals | self.rhs.requires

        return vals

    @property
    def produces(self):
        return None

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

    def evaluate(self, data: dict[str, np.ndarray], index: DataIndex) -> np.ndarray:
        lhs: np.typing.ArrayLike
        rhs: Optional[np.typing.ArrayLike]
        match self.lhs:
            case DerivedColumn():
                lhs = self.lhs.evaluate(data, index)
            case Column():
                lhs = data[self.lhs.column_name]
            case _:
                lhs = self.lhs
        match self.rhs:
            case DerivedColumn():
                rhs = self.rhs.evaluate(data, index)
            case Column():
                rhs = data[self.rhs.column_name]
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
        strategy: EvaluateStrategy = EvaluateStrategy.ROW_WISE,
        description: Optional[str] = None,
        **kwargs: Any,
    ):
        self.__func = func
        self.__requires = requires
        self.__kwargs = kwargs
        self.__produces = produces
        self.__format = format
        self.__strategy = strategy
        self.description = description

    @property
    def requires(self):
        return copy(self.__requires)

    @property
    def produces(self):
        return copy(self.__produces)

    @property
    def signature(self):
        return signature(self.__func)

    @property
    def kwarg_names(self):
        return self.__kwargs.keys()

    def get_units(self, units: dict[str, np.ndarray]):
        test_data: dict[str, Any]
        match self.__strategy:
            case EvaluateStrategy.ROW_WISE:
                test_data = {
                    name: np.random.randint(20, 40) for name in self.__requires
                }
            case _:
                test_data = {
                    name: np.random.randint(20, 40, 2) for name in self.__requires
                }

        test_data = {
            name: td * units[name] if units.get(name) is not None else td
            for name, td in test_data.items()
        }

        results = self.__func(**test_data, **self.__kwargs)
        if not isinstance(results, dict):
            results = {self.__func.__name__: results}

        return {
            name: result.unit if isinstance(result, u.Quantity) else None
            for name, result in results.items()
        }

    def evaluate(self, data: dict[str, np.ndarray], index: DataIndex):
        data = {name: data[name] for name in self.__requires}
        match self.__strategy:
            case EvaluateStrategy.VECTORIZE:
                return evaluate_vectorized(data, self.__func, self.__kwargs)
            case EvaluateStrategy.ROW_WISE:
                return evaluate_rows(data, self.__func, self.__kwargs)
            case EvaluateStrategy.CHUNKED:
                if not isinstance(index, tuple):
                    raise ValueError(
                        "Cannot evaluate in CHUNKED strategy with a non-chunked index"
                    )
                return evaluate_chunks(data, self.__func, self.__kwargs, index)

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
        column_name: str,
        value: float | u.Quantity,
        operator: Callable[[table.Column, float | u.Quantity], np.ndarray],
    ):
        self.column_name = column_name
        self.value = value
        self.operator = operator

    @property
    def requires(self):
        return {self.column_name}

    def apply(self, column: u.Quantity | np.ndarray) -> np.ndarray:
        """
        mask the dataset based on the mask.
        """
        # Astropy's errors are good enough here
        if isinstance(column, table.Table):
            column = column[self.column_name]

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
