"""Tests for converting live column expressions into serialized ones."""

import astropy.units as u
import numpy as np
import opencosmo as oc
import pytest
from opencosmo.serde import (
    ArithmeticOperator,
    ColumnReference,
    ComparisonMask,
    CompoundMask,
    BooleanOperator,
    Expression,
    Mask,
    MembershipMask,
    NumberLiteral,
    QuantileExpression,
    QuantityLiteral,
    ReductionExpression,
    SerdeError,
    UnaryExpression,
    serialize_column_expression,
    serialize_column_mask,
)
from opencosmo.serde.rehydrate import expression as live
from pydantic import TypeAdapter

col = oc.col


def __round_trip(expression):
    serialized = serialize_column_expression(expression)
    assert not isinstance(serialized, SerdeError), serialized
    # Must survive a JSON round trip and rebuild to an equal serialization
    revalidated = TypeAdapter(Expression).validate_json(
        TypeAdapter(Expression).dump_json(serialized)
    )
    assert revalidated == serialized
    again = serialize_column_expression(live.expression_to_live(serialized))
    assert again == serialized
    return serialized


def test_column_and_literals():
    assert serialize_column_expression(col("mass")) == ColumnReference(name="mass")
    assert serialize_column_expression("mass") == ColumnReference(name="mass")
    assert serialize_column_expression(3) == NumberLiteral(value=3)
    assert serialize_column_expression(np.float64(2.5)) == NumberLiteral(value=2.5)
    assert serialize_column_expression(2 * u.Msun) == QuantityLiteral(
        value=2, unit="solMass"
    )
    assert serialize_column_expression(3 * u.dimensionless_unscaled) == NumberLiteral(
        value=3
    )


@pytest.mark.parametrize(
    "expression",
    [
        col("a") + col("b"),
        col("a") - 2,
        3 * col("a"),
        col("a") / col("b"),
        col("a") ** 2,
        col("a").log10(),
        col("a").exp10(),
        col("a").sqrt(),
        col("a").arcsin(),
        col("a").arccos(),
        col("a").arctan2(col("b")),
        col("a").mean(),
        col("a").std(),
        col("a").var(),
        col("a").min(),
        col("a").max(),
        col("a").median(),
        col("a").sum(),
        col("a").quantile(0.25),
        (col("a") - col("a").mean()) / col("a").std(),
        col("a") * (2 * u.km),
    ],
)
def test_expression_round_trips(expression):
    __round_trip(expression)


def test_arithmetic_structure():
    result = serialize_column_expression(col("a") / 2)
    assert result.operator is ArithmeticOperator.DIVIDE
    assert result.left == ColumnReference(name="a")
    assert result.right == NumberLiteral(value=2)


def test_reduction_and_quantile_structure():
    assert isinstance(serialize_column_expression(col("a").mean()), ReductionExpression)
    result = serialize_column_expression(col("a").quantile(1))
    assert isinstance(result, QuantileExpression)
    assert result.quantile == 1.0
    assert isinstance(serialize_column_expression(col("a").log10()), UnaryExpression)


@pytest.mark.parametrize(
    "mask",
    [
        col("a") > 1,
        col("a") >= 1,
        col("a") < 1,
        col("a") <= 1,
        col("a") == 1,
        col("a") != 1,
        col("a") > 1 * u.Msun,
        col("a").isin([1, 2, 3]),
        col("a").isin(np.array([1.0, 2.0])),
        (col("a") > 1) & (col("b") < 2),
        ((col("a") > 1) | (col("b") < 2)) & (col("c") == 3),
        col("a") > col("b").mean(),
    ],
)
def test_mask_round_trips(mask):
    serialized = serialize_column_mask(mask)
    assert not isinstance(serialized, SerdeError), serialized
    adapter = TypeAdapter(Mask)
    assert adapter.validate_json(adapter.dump_json(serialized)) == serialized
    assert serialize_column_mask(live.mask_to_live(serialized)) == serialized


def test_mask_structure():
    result = serialize_column_mask((col("a") > 1) | (col("b").isin([1, 2])))
    assert isinstance(result, CompoundMask)
    assert result.operator is BooleanOperator.OR
    assert isinstance(result.left, ComparisonMask)
    assert isinstance(result.right, MembershipMask)
    anded = serialize_column_mask((col("a") > 1) & (col("b") > 1))
    assert anded.operator is BooleanOperator.AND


@pytest.mark.parametrize(
    "value",
    [
        object(),
        True,
        np.array([1.0, 2.0]) * u.km,
        col("a").log10(u.MagUnit),
        [1, 2],
    ],
)
def test_unsupported_expressions_return_errors(value):
    result = serialize_column_expression(value)
    assert isinstance(result, SerdeError)
    assert result.operation == "serialize_column_expression"


def test_unsupported_mask_returns_error():
    result = serialize_column_mask(object())
    assert isinstance(result, SerdeError)
    assert result.operation == "serialize_column_mask"
