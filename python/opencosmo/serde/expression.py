"""Validated wire models for column expressions and masks."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

import astropy.units as u
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
)


class ExpressionModel(BaseModel):
    """Base configuration for serialized expressions and masks."""

    model_config = ConfigDict(frozen=True, extra="forbid")


type FiniteNumber = Annotated[
    StrictInt | StrictFloat,
    Field(allow_inf_nan=False),
]


class ArithmeticOperator(StrEnum):
    """Supported binary column arithmetic operations."""

    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    POWER = "power"


class UnaryOperator(StrEnum):
    """Supported unary column operations."""

    LOG10 = "log10"
    EXP10 = "exp10"
    SQRT = "sqrt"
    ARCSIN = "arcsin"
    ARCCOS = "arccos"


class ReductionOperator(StrEnum):
    """Supported scalar column reductions without additional arguments."""

    MEAN = "mean"
    STANDARD_DEVIATION = "std"
    VARIANCE = "var"
    MINIMUM = "min"
    MAXIMUM = "max"
    MEDIAN = "median"
    SUM = "sum"


class ComparisonOperator(StrEnum):
    """Supported column comparison operations."""

    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"


class BooleanOperator(StrEnum):
    """Supported operations for combining masks."""

    AND = "and"
    OR = "or"


class ColumnReference(ExpressionModel):
    """A reference to a visible column by name."""

    kind: Literal["column"] = "column"
    name: str = Field(min_length=1)


class NumberLiteral(ExpressionModel):
    """A finite, unitless scalar value."""

    kind: Literal["number"] = "number"
    value: FiniteNumber


class QuantityLiteral(ExpressionModel):
    """A finite scalar value with an Astropy unit."""

    kind: Literal["quantity"] = "quantity"
    value: FiniteNumber
    unit: str = Field(min_length=1)

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, value: str) -> str:
        """Validate and normalize an Astropy unit string."""
        try:
            return u.Unit(value).to_string()
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid Astropy unit: {value!r}") from error


type ScalarLiteral = Annotated[
    NumberLiteral | QuantityLiteral,
    Field(discriminator="kind"),
]


class ArithmeticExpression(ExpressionModel):
    """A binary arithmetic expression."""

    kind: Literal["arithmetic"] = "arithmetic"
    operator: ArithmeticOperator
    left: Expression
    right: Expression


class UnaryExpression(ExpressionModel):
    """A unary mathematical expression."""

    kind: Literal["unary"] = "unary"
    operator: UnaryOperator
    operand: Expression


class Arctan2Expression(ExpressionModel):
    """A two-argument arctangent expression."""

    kind: Literal["arctan2"] = "arctan2"
    left: Expression
    right: Expression


class ReductionExpression(ExpressionModel):
    """A scalar reduction of an expression."""

    kind: Literal["reduction"] = "reduction"
    operator: ReductionOperator
    operand: Expression


class QuantileExpression(ExpressionModel):
    """A quantile reduction of an expression."""

    kind: Literal["quantile"] = "quantile"
    operand: Expression
    quantile: Annotated[StrictFloat, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


type Expression = Annotated[
    ColumnReference
    | NumberLiteral
    | QuantityLiteral
    | ArithmeticExpression
    | UnaryExpression
    | Arctan2Expression
    | ReductionExpression
    | QuantileExpression,
    Field(discriminator="kind"),
]


class ComparisonMask(ExpressionModel):
    """A comparison between two expressions."""

    kind: Literal["comparison"] = "comparison"
    operator: ComparisonOperator
    left: Expression
    right: Expression


class MembershipMask(ExpressionModel):
    """A test for membership in a collection of scalar literals."""

    kind: Literal["membership"] = "membership"
    operand: Expression
    values: tuple[ScalarLiteral, ...]


class CompoundMask(ExpressionModel):
    """A boolean combination of two masks."""

    kind: Literal["compound"] = "compound"
    operator: BooleanOperator
    left: Mask
    right: Mask


type Mask = Annotated[
    ComparisonMask | MembershipMask | CompoundMask,
    Field(discriminator="kind"),
]


ArithmeticExpression.model_rebuild(_types_namespace={"Expression": Expression})
UnaryExpression.model_rebuild(_types_namespace={"Expression": Expression})
Arctan2Expression.model_rebuild(_types_namespace={"Expression": Expression})
ReductionExpression.model_rebuild(_types_namespace={"Expression": Expression})
QuantileExpression.model_rebuild(_types_namespace={"Expression": Expression})
ComparisonMask.model_rebuild(_types_namespace={"Expression": Expression})
MembershipMask.model_rebuild(_types_namespace={"Expression": Expression})
CompoundMask.model_rebuild(_types_namespace={"Mask": Mask})


__all__ = [
    "ArithmeticExpression",
    "ArithmeticOperator",
    "Arctan2Expression",
    "BooleanOperator",
    "ColumnReference",
    "ComparisonMask",
    "ComparisonOperator",
    "CompoundMask",
    "Expression",
    "ExpressionModel",
    "FiniteNumber",
    "Mask",
    "MembershipMask",
    "NumberLiteral",
    "QuantileExpression",
    "QuantityLiteral",
    "ReductionExpression",
    "ReductionOperator",
    "ScalarLiteral",
    "UnaryExpression",
    "UnaryOperator",
]
