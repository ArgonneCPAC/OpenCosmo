"""Rehydrate serialized column expressions and masks."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import astropy.units as u

from opencosmo.column.column import (
    Column,
    ColumnMask,
    CompoundColumnMask,
    DerivedScalarValue,
    col,
)

from ..expression import (
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

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..expression import Expression, Mask

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
            return _arithmetic(
                operation,
                expression_to_live(left),
                expression_to_live(right),
            )
        case UnaryExpression(operator=operation, operand=operand):
            live_operand = expression_to_live(operand)
            if not isinstance(live_operand, Column):
                raise ValueError(f"{operation.value} requires a vector expression")
            return _unary(operation, live_operand)
        case Arctan2Expression(left=left, right=right):
            live_left = expression_to_live(left)
            if not isinstance(live_left, Column):
                raise ValueError("arctan2 requires a vector expression on the left")
            return live_left.arctan2(expression_to_live(right))
        case ReductionExpression(operator=operation, operand=operand):
            live_operand = expression_to_live(operand)
            if not isinstance(live_operand, Column):
                raise ValueError(f"{operation.value} requires a vector expression")
            return _reduce(operation, live_operand)
        case QuantileExpression(operand=operand, quantile=quantile):
            live_operand = expression_to_live(operand)
            if not isinstance(live_operand, Column):
                raise ValueError("quantile requires a vector expression")
            return live_operand.quantile(quantile)
    raise TypeError(f"Unsupported expression type: {type(expression).__name__}")


def mask_to_live(mask: Mask) -> LiveMask:
    """Build a lazy OpenCosmo mask from a serialized mask."""
    match mask:
        case ComparisonMask(operator=operation, left=left, right=right):
            live_left = expression_to_live(left)
            live_right = expression_to_live(right)
            if not isinstance(live_left, Column) and not isinstance(live_right, Column):
                raise ValueError("A comparison mask requires a vector expression")
            return ColumnMask(
                live_left,  # type: ignore[arg-type]
                live_right,  # type: ignore[arg-type]
                _comparison_operator(operation),  # type: ignore[arg-type]
            )
        case MembershipMask(operand=operand, values=values):
            live_operand = expression_to_live(operand)
            if not isinstance(live_operand, Column):
                raise ValueError("A membership mask requires a vector expression")
            live_values = tuple(expression_to_live(value) for value in values)
            return live_operand.isin(live_values)
        case CompoundMask(operator=operation, left=left, right=right):
            live_left = mask_to_live(left)
            live_right = mask_to_live(right)
            if operation is BooleanOperator.AND:
                return live_left & live_right
            return live_left | live_right
    raise TypeError(f"Unsupported mask type: {type(mask).__name__}")


def _arithmetic(
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


def _unary(operation: UnaryOperator, operand: Column) -> Column:
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


def _reduce(operation: ReductionOperator, operand: Column) -> DerivedScalarValue:
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


def _comparison_operator(
    operation: ComparisonOperator,
) -> Callable[[LiveExpression, LiveExpression], bool]:
    operations: dict[
        ComparisonOperator, Callable[[LiveExpression, LiveExpression], bool]
    ] = {
        ComparisonOperator.EQUAL: operator.eq,
        ComparisonOperator.NOT_EQUAL: operator.ne,
        ComparisonOperator.GREATER_THAN: operator.gt,
        ComparisonOperator.GREATER_THAN_OR_EQUAL: operator.ge,
        ComparisonOperator.LESS_THAN: operator.lt,
        ComparisonOperator.LESS_THAN_OR_EQUAL: operator.le,
    }
    return operations[operation]


__all__ = [
    "LiveExpression",
    "LiveMask",
    "LiveScalar",
    "expression_to_live",
    "mask_to_live",
]
