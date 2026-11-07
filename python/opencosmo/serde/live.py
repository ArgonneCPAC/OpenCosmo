"""Convert validated wire values into live OpenCosmo objects."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np

from opencosmo.column.column import (
    Column,
    ColumnMask,
    CompoundColumnMask,
    DerivedScalarValue,
    col,
)
from opencosmo.spatial.builders import make_box, make_cone, make_skybox
from opencosmo.spatial.region import HealpixRegion

from .expression import (
    Arctan2Expression,
    ArithmeticExpression,
    ArithmeticOperator,
    BooleanOperator,
    ColumnReference,
    ComparisonMask,
    ComparisonOperator,
    CompoundMask,
    MembershipMask,
    NumberLiteral,
    QuantileExpression,
    QuantityLiteral,
    ReductionExpression,
    ReductionOperator,
    UnaryExpression,
    UnaryOperator,
)
from .region import (
    BoxRegionMessage,
    ConeRegionMessage,
    HealpixRegionMessage,
    SkyboxRegionMessage,
)

if TYPE_CHECKING:
    from opencosmo.column.column import MaskOperator
    from opencosmo.spatial.protocols import Region

    from .expression import Expression, Mask, ScalarLiteral
    from .region import RegionMessage

type LiveScalar = int | float | u.Quantity
type LiveExpression = Column | DerivedScalarValue | LiveScalar
type LiveMask = ColumnMask | CompoundColumnMask


def expression_to_live(expression: Expression) -> LiveExpression:
    """Build a lazy OpenCosmo expression from a serialized expression."""
    match expression:
        case ColumnReference(name=name):
            return col(name)
        case NumberLiteral(value=value):
            return value
        case QuantityLiteral(value=value, unit=unit):
            return u.Quantity(value, u.Unit(unit))
        case ArithmeticExpression(operator=operation, left=left, right=right):
            return __arithmetic(
                operation,
                expression_to_live(left),
                expression_to_live(right),
            )
        case UnaryExpression(operator=operation, operand=operand):
            live_operand = expression_to_live(operand)
            if not isinstance(live_operand, Column):
                raise TypeError(f"{operation.value} requires a vector expression")
            return __unary(operation, live_operand)
        case Arctan2Expression(left=left, right=right):
            live_left = expression_to_live(left)
            if not isinstance(live_left, Column):
                raise TypeError("arctan2 requires a vector expression on the left")
            return live_left.arctan2(expression_to_live(right))
        case ReductionExpression(operator=operation, operand=operand):
            live_operand = expression_to_live(operand)
            if not isinstance(live_operand, Column):
                raise TypeError(f"{operation.value} requires a vector expression")
            return __reduce(operation, live_operand)
        case QuantileExpression(operand=operand, quantile=quantile):
            live_operand = expression_to_live(operand)
            if not isinstance(live_operand, Column):
                raise TypeError("quantile requires a vector expression")
            return live_operand.quantile(quantile)
    raise TypeError(f"Unsupported expression type: {type(expression).__name__}")


def mask_to_live(mask: Mask) -> LiveMask:
    """Build a lazy OpenCosmo mask from a serialized mask."""
    match mask:
        case ComparisonMask(operator=operation, left=left, right=right):
            live_left = expression_to_live(left)
            live_right = expression_to_live(right)
            if not isinstance(live_left, Column) and not isinstance(live_right, Column):
                raise TypeError("A comparison mask requires a vector expression")
            return ColumnMask(live_left, live_right, __comparison_operator(operation))
        case MembershipMask(operand=operand, values=values):
            live_operand = expression_to_live(operand)
            if not isinstance(live_operand, Column):
                raise TypeError("A membership mask requires a vector expression")
            return live_operand.isin(tuple(__scalar_to_live(value) for value in values))
        case CompoundMask(operator=operation, left=left, right=right):
            live_left = mask_to_live(left)
            live_right = mask_to_live(right)
            if operation is BooleanOperator.AND:
                return live_left & live_right
            return live_left | live_right
    raise TypeError(f"Unsupported mask type: {type(mask).__name__}")


def __arithmetic(
    operation: ArithmeticOperator,
    left: LiveExpression,
    right: LiveExpression,
) -> LiveExpression:
    operations = {
        ArithmeticOperator.ADD: operator.add,
        ArithmeticOperator.SUBTRACT: operator.sub,
        ArithmeticOperator.MULTIPLY: operator.mul,
        ArithmeticOperator.DIVIDE: operator.truediv,
        ArithmeticOperator.POWER: operator.pow,
    }
    function = operations[operation]
    if isinstance(left, Column) or isinstance(right, Column):
        return Column(left, right, function)
    if isinstance(left, DerivedScalarValue) or isinstance(right, DerivedScalarValue):
        return DerivedScalarValue(left, right, function)
    return function(left, right)


def __unary(operation: UnaryOperator, operand: Column) -> Column:
    match operation:
        case UnaryOperator.LOG10:
            return operand.log10()
        case UnaryOperator.EXP10:
            return operand.exp10()
        case UnaryOperator.SQRT:
            return operand.sqrt()
        case UnaryOperator.ARCSIN:
            return operand.arcsin()
        case UnaryOperator.ARCCOS:
            return operand.arccos()


def __reduce(operation: ReductionOperator, operand: Column) -> DerivedScalarValue:
    match operation:
        case ReductionOperator.MEAN:
            return operand.mean()
        case ReductionOperator.STANDARD_DEVIATION:
            return operand.std()
        case ReductionOperator.VARIANCE:
            return operand.var()
        case ReductionOperator.MINIMUM:
            return operand.min()
        case ReductionOperator.MAXIMUM:
            return operand.max()
        case ReductionOperator.MEDIAN:
            return operand.median()
        case ReductionOperator.SUM:
            return operand.sum()


def __scalar_to_live(value: ScalarLiteral) -> LiveScalar:
    match value:
        case NumberLiteral(value=number):
            return number
        case QuantityLiteral(value=number, unit=unit):
            return u.Quantity(number, u.Unit(unit))


def __comparison_operator(operation: ComparisonOperator) -> MaskOperator:
    operations: dict[ComparisonOperator, MaskOperator] = {
        ComparisonOperator.EQUAL: operator.eq,
        ComparisonOperator.NOT_EQUAL: operator.ne,
        ComparisonOperator.GREATER_THAN: operator.gt,
        ComparisonOperator.GREATER_THAN_OR_EQUAL: operator.ge,
        ComparisonOperator.LESS_THAN: operator.lt,
        ComparisonOperator.LESS_THAN_OR_EQUAL: operator.le,
    }
    return operations[operation]


def region_to_live(region: RegionMessage) -> Region:
    """Build an OpenCosmo region from a serialized region message."""
    match region:
        case BoxRegionMessage(p1=p1, p2=p2):
            return make_box(p1, p2)
        case ConeRegionMessage(center=center, radius=radius):
            return make_cone(center, radius)
        case SkyboxRegionMessage(p1=p1, p2=p2):
            return make_skybox(p1, p2)
        case HealpixRegionMessage(pixels=pixels, nside=nside):
            return HealpixRegion(np.asarray(sorted(pixels), dtype=np.int64), nside)
    raise TypeError(f"Unsupported region type: {type(region).__name__}")


def to_column(expression: Expression) -> Column:
    """Build a lazy column, rejecting expressions that do not produce one."""
    live = expression_to_live(expression)
    if not isinstance(live, Column):
        raise TypeError("Expression must produce a column")
    return live


def to_column_or_scalar(expression: Expression) -> Column | DerivedScalarValue:
    """Build a lazy column or scalar reduction, rejecting plain literals."""
    live = expression_to_live(expression)
    if not isinstance(live, Column | DerivedScalarValue):
        raise TypeError("Expression must produce a column or a scalar reduction")
    return live


def to_unit(unit: str) -> u.UnitBase:
    """Build an Astropy unit from a normalized unit string."""
    return u.Unit(unit)


def to_rows(rows: tuple[int, ...]) -> np.ndarray:
    """Build a signed 64-bit row index from row positions."""
    return np.asarray(rows, dtype=np.int64)


__all__ = [
    "LiveExpression",
    "LiveMask",
    "LiveScalar",
    "expression_to_live",
    "mask_to_live",
    "region_to_live",
    "to_column",
    "to_column_or_scalar",
    "to_rows",
    "to_unit",
]
