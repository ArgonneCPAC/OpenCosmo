"""Convert live OpenCosmo column expressions into their serialized counterparts."""

from __future__ import annotations

import operator
from functools import partial
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np

from opencosmo.column.column import (
    Column,
    ColumnMask,
    CompoundColumnMask,
    DerivedScalarValue,
    ident,
    render_op,
)

from .errors import make_serde_error
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

if TYPE_CHECKING:
    from .errors import SerdeError
    from .expression import Expression, Mask, ScalarLiteral
    from .rehydrate.expression import LiveExpression, LiveMask

type SerializeExpressionResponse = Expression | SerdeError
type SerializeMaskResponse = Mask | SerdeError

ARITHMETIC = {
    "+": ArithmeticOperator.ADD,
    "-": ArithmeticOperator.SUBTRACT,
    "*": ArithmeticOperator.MULTIPLY,
    "/": ArithmeticOperator.DIVIDE,
    "**": ArithmeticOperator.POWER,
}
UNARY = {
    "log10": UnaryOperator.LOG10,
    "exp10": UnaryOperator.EXP10,
    "sqrt": UnaryOperator.SQRT,
    "arcsin": UnaryOperator.ARCSIN,
    "arccos": UnaryOperator.ARCCOS,
}
REDUCTIONS = {
    "mean": ReductionOperator.MEAN,
    "std": ReductionOperator.STANDARD_DEVIATION,
    "var": ReductionOperator.VARIANCE,
    "min": ReductionOperator.MINIMUM,
    "max": ReductionOperator.MAXIMUM,
    "median": ReductionOperator.MEDIAN,
    "sum": ReductionOperator.SUM,
}
COMPARISONS = {
    operator.eq: ComparisonOperator.EQUAL,
    operator.ne: ComparisonOperator.NOT_EQUAL,
    operator.gt: ComparisonOperator.GREATER_THAN,
    operator.ge: ComparisonOperator.GREATER_THAN_OR_EQUAL,
    operator.lt: ComparisonOperator.LESS_THAN,
    operator.le: ComparisonOperator.LESS_THAN_OR_EQUAL,
}


def __scalar(value: int | float | u.Quantity) -> NumberLiteral | QuantityLiteral:
    if isinstance(value, u.Quantity):
        if not value.isscalar:
            raise ValueError(
                f"Only scalar quantities can be serialized, got shape {value.shape}"
            )
        if value.unit == u.dimensionless_unscaled:
            return NumberLiteral(value=value.value.item())
        return QuantityLiteral(value=value.value.item(), unit=value.unit.to_string())
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise TypeError(f"Unsupported scalar type: {type(value).__name__}")
    return NumberLiteral(value=value.item() if isinstance(value, np.generic) else value)


def __check_unit_container(operation: object) -> None:
    if isinstance(operation, partial):
        for container in operation.keywords.values():
            if container is not u.DexUnit:
                raise ValueError("Only dex logarithmic units can be serialized")


def __node(node: Column | DerivedScalarValue) -> Expression:
    operation = node.operation
    name = render_op(operation)
    left = expression_from_live(node.lhs)  # type: ignore[arg-type]
    if node.rhs is None:
        if name in UNARY:
            __check_unit_container(operation)
            return UnaryExpression(operator=UNARY[name], operand=left)
        if name in REDUCTIONS:
            return ReductionExpression(operator=REDUCTIONS[name], operand=left)
        if name == "quantile" and isinstance(operation, partial):
            return QuantileExpression(
                operand=left, quantile=float(operation.keywords["q"])
            )
    else:
        right = expression_from_live(node.rhs)  # type: ignore[arg-type]
        if name in ARITHMETIC:
            return ArithmeticExpression(
                operator=ARITHMETIC[name], left=left, right=right
            )
        if name == "arctan2":
            return Arctan2Expression(left=left, right=right)
    raise ValueError(f"Unsupported column operation: {name or operation!r}")


def expression_from_live(expression: LiveExpression | str) -> Expression:
    """Build a serialized expression from a lazy OpenCosmo expression."""
    match expression:
        case str():
            return ColumnReference(name=expression)
        case Column() if expression.operation is ident:
            return expression_from_live(expression.lhs)  # type: ignore[arg-type]
        case Column() | DerivedScalarValue():
            return __node(expression)
        case _:
            return __scalar(expression)


def mask_from_live(mask: LiveMask) -> Mask:
    """Build a serialized mask from a lazy OpenCosmo mask."""
    match mask:
        case ColumnMask():
            operand = expression_from_live(mask.left)  # type: ignore[arg-type]
            if mask.operator is np.isin:
                values: tuple[ScalarLiteral, ...] = tuple(
                    __scalar(value)
                    for value in mask.right  # type: ignore[union-attr]
                )
                return MembershipMask(operand=operand, values=values)
            comparison = COMPARISONS.get(mask.operator)  # type: ignore[call-overload]
            if comparison is None:
                raise ValueError("Unsupported mask comparison")
            right = expression_from_live(mask.right)  # type: ignore[arg-type]
            return ComparisonMask(operator=comparison, left=operand, right=right)
        case CompoundColumnMask():
            # The combining function is an opaque lambda: probe it to tell & from |
            combined = mask.op(np.array([True]), np.array([False]))[0]
            return CompoundMask(
                operator=BooleanOperator.OR if combined else BooleanOperator.AND,
                left=mask_from_live(mask.left),
                right=mask_from_live(mask.right),
            )
    raise TypeError(f"Unsupported mask type: {type(mask).__name__}")


def serialize_column_expression(
    expression: LiveExpression | str,
) -> SerializeExpressionResponse:
    """Serialize a column expression, returning a structured error on failure.

    Parameters
    ----------
    expression : Column, DerivedScalarValue, str, int, float or Quantity
        An expression built from :py:func:`opencosmo.col` and scalars. A string
        is treated as a column name.

    Returns
    -------
    Expression or SerdeError
        The serialized expression, or an error if it cannot be represented
        (for example, an evaluated user function or a non-dex log unit).
    """
    try:
        return expression_from_live(expression)
    except Exception as error:
        return make_serde_error(
            "serialize_column_expression",
            error,
            input_type=type(expression).__name__,
        )


def serialize_column_mask(mask: LiveMask) -> SerializeMaskResponse:
    """Serialize a column mask, returning a structured error on failure."""
    try:
        return mask_from_live(mask)
    except Exception as error:
        return make_serde_error(
            "serialize_column_mask",
            error,
            input_type=type(mask).__name__,
        )


__all__ = [
    "SerializeExpressionResponse",
    "SerializeMaskResponse",
    "serialize_column_expression",
    "serialize_column_mask",
]
