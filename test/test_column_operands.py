"""Pure tests for expression operand contracts. No test data required."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from opencosmo.column.column import Column, ColumnMask, DerivedScalarValue, col
from opencosmo.serde import (
    ArithmeticExpression,
    ArithmeticOperator,
    ColumnReference,
    ComparisonMask,
    ComparisonOperator,
    MembershipMask,
    NumberLiteral,
    QuantityLiteral,
    ReductionExpression,
    ReductionOperator,
)
from opencosmo.serde.live import expression_to_live, mask_to_live

DATA = {"x": np.array([1.0, 2.0, 3.0, 6.0]), "tag": np.array([1, 2, 3, 4])}


def _mean(name: str) -> ReductionExpression:
    return ReductionExpression(
        operator=ReductionOperator.MEAN, operand=ColumnReference(name=name)
    )


class TestScalarReductionOperands:
    def test_column_minus_reduction(self):
        expr = col("x") - col("x").mean()
        assert isinstance(expr, Column)
        np.testing.assert_allclose(expr.evaluate(DATA), [-2.0, -1.0, 0.0, 3.0])

    def test_reduction_on_left_produces_column(self):
        expr = col("x").mean() - col("x")
        assert isinstance(expr, Column)
        assert isinstance(expr.lhs, DerivedScalarValue)
        np.testing.assert_allclose(expr.evaluate(DATA), [2.0, 1.0, 0.0, -3.0])

    def test_reduction_arithmetic_stays_scalar(self):
        expr = col("x").max() - col("x").min()
        assert isinstance(expr, DerivedScalarValue)
        assert expr.evaluate(DATA) == pytest.approx(5.0)

    def test_normalization(self):
        expr = (col("x") - col("x").mean()) / col("x").std()
        result = expr.evaluate(DATA)
        assert np.mean(result) == pytest.approx(0.0)
        assert np.std(result) == pytest.approx(1.0)

    def test_arctan2_accepts_reduction(self):
        expr = col("x").arctan2(col("x").max())
        np.testing.assert_allclose(expr.evaluate(DATA), np.arctan2(DATA["x"], 6.0))

    def test_non_scalar_quantity_still_rejected(self):
        with pytest.raises(ValueError, match="Only scalar Quantity values"):
            col("x") * (np.ones(2) * u.m)
        with pytest.raises(ValueError, match="Only scalar Quantity values"):
            col("x").mean() * (np.ones(2) * u.m)


class TestComparisonOperands:
    def test_column_compared_with_reduction(self):
        mask = col("x") > col("x").mean()
        np.testing.assert_array_equal(mask.apply(DATA), [False, False, False, True])

    def test_reduction_compared_with_column(self):
        mask = col("x").mean() < col("x")
        np.testing.assert_array_equal(mask.apply(DATA), [False, False, False, True])

    def test_columns_compared(self):
        mask = col("x") >= col("tag")
        np.testing.assert_array_equal(mask.apply(DATA), [True, True, True, True])

    def test_isin_keeps_supplied_values(self):
        values = np.array([2, 4])
        mask = col("tag").isin(values)
        assert isinstance(mask, ColumnMask)
        assert mask.right is values
        np.testing.assert_array_equal(mask.apply(DATA), [False, True, False, True])


class TestSerdeOperands:
    def test_arithmetic_with_reduction(self):
        expression = ArithmeticExpression(
            operator=ArithmeticOperator.SUBTRACT,
            left=ColumnReference(name="x"),
            right=_mean("x"),
        )
        live = expression_to_live(expression)
        assert isinstance(live, Column)
        np.testing.assert_allclose(live.evaluate(DATA), [-2.0, -1.0, 0.0, 3.0])

    def test_comparison_with_reduction(self):
        mask = ComparisonMask(
            operator=ComparisonOperator.GREATER_THAN,
            left=ColumnReference(name="x"),
            right=_mean("x"),
        )
        live = mask_to_live(mask)
        np.testing.assert_array_equal(live.apply(DATA), [False, False, False, True])

    def test_comparison_requires_vector_expression(self):
        mask = ComparisonMask(
            operator=ComparisonOperator.GREATER_THAN,
            left=_mean("x"),
            right=NumberLiteral(value=1),
        )
        with pytest.raises(TypeError, match="requires a vector expression"):
            mask_to_live(mask)

    def test_membership_converts_scalar_literals(self):
        mask = MembershipMask(
            operand=ColumnReference(name="tag"),
            values=(NumberLiteral(value=2), NumberLiteral(value=4)),
        )
        live = mask_to_live(mask)
        np.testing.assert_array_equal(live.apply(DATA), [False, True, False, True])

    def test_membership_converts_quantity_literals(self):
        mask = MembershipMask(
            operand=ColumnReference(name="x"),
            values=(QuantityLiteral(value=2.0, unit="m"),),
        )
        live = mask_to_live(mask)
        assert isinstance(live, ColumnMask)
        assert live.right == (2.0 * u.m,)
