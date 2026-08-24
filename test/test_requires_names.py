"""Unit tests for the requires_names property on column types."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
from opencosmo.column.column import (
    EvaluatedColumn,
    RawColumn,
    col,
)
from opencosmo.column.evaluate import EvaluateStrategy

# ---------------------------------------------------------------------------
# Column
# ---------------------------------------------------------------------------


class TestColumnRequiresNames:
    def test_simple_leaf_unbound(self):
        c = col("fof_halo_mass")
        assert c.requires_names == {"fof_halo_mass"}

    def test_binary_expression_unbound(self):
        expr = col("fof_halo_mass") * col("fof_halo_com_vx")
        assert expr.requires_names == {"fof_halo_mass", "fof_halo_com_vx"}

    def test_chained_expression_unbound(self):
        expr = col("a") * col("b") + col("c")
        assert expr.requires_names == {"a", "b", "c"}

    def test_scalar_operand_not_included(self):
        # Multiplying by a plain int/float should not add any name
        expr = col("fof_halo_mass") * 5
        assert expr.requires_names == {"fof_halo_mass"}

    def test_returns_fresh_set(self):
        c = col("fof_halo_mass")
        s1 = c.requires_names
        s2 = c.requires_names
        assert s1 == s2
        s1.add("extra")
        assert "extra" not in c.requires_names

    def test_bound_column_same_result(self):
        expr = col("fof_halo_mass") * col("fof_halo_com_vx")
        name_to_uuid = {
            "fof_halo_mass": uuid4(),
            "fof_halo_com_vx": uuid4(),
        }
        expr.name = "output"
        bound = expr.bind(name_to_uuid, set(name_to_uuid.values()))
        assert bound.requires_names == {"fof_halo_mass", "fof_halo_com_vx"}

    def test_duplicate_names_deduplicated(self):
        # Same column used twice in an expression
        m = col("fof_halo_mass")
        expr = m * m
        assert expr.requires_names == {"fof_halo_mass"}


# ---------------------------------------------------------------------------
# RawColumn
# ---------------------------------------------------------------------------


class TestRawColumnRequiresNames:
    def test_plain_raw_column_empty(self):
        rc = RawColumn("fof_halo_mass", "halo mass", uuid4())
        assert rc.requires_names == set()

    def test_alias_unbound_returns_underlying_name(self):
        rc = RawColumn("fof_halo_mass", "halo mass", uuid4(), alias="mass")
        assert rc.requires_names == {"fof_halo_mass"}

    def test_alias_bound_returns_underlying_name(self):
        dep_uuid = uuid4()
        rc = RawColumn(
            "fof_halo_mass",
            "halo mass",
            uuid4(),
            alias="mass",
            _dep_uuid=dep_uuid,
        )
        assert rc.requires_names == {"fof_halo_mass"}

    def test_alias_bind_then_requires_names(self):
        rc = RawColumn("fof_halo_mass", "halo mass", uuid4(), alias="mass")
        name_to_uuid = {"fof_halo_mass": uuid4()}
        bound = rc.bind(name_to_uuid, set(name_to_uuid.values()))
        assert bound.requires_names == {"fof_halo_mass"}

    def test_plain_returns_fresh_set(self):
        rc = RawColumn("fof_halo_mass", "halo mass", uuid4())
        s = rc.requires_names
        s.add("extra")
        assert rc.requires_names == set()


# ---------------------------------------------------------------------------
# DerivedScalarValue
# ---------------------------------------------------------------------------


class TestDerivedScalarValueRequiresNames:
    def test_simple_mean_unbound(self):
        scalar = col("fof_halo_mass").mean()
        assert scalar.requires_names == {"fof_halo_mass"}

    def test_compound_scalar_arithmetic_unbound(self):
        m = col("fof_halo_mass")
        # (m - m.mean()) / m.std() — the outer Column wraps two DerivedScalarValues
        # but the scalar itself (m.mean()) should only see "fof_halo_mass"
        scalar = m.mean()
        assert scalar.requires_names == {"fof_halo_mass"}

    def test_scalar_arithmetic_between_scalars(self):
        m = col("fof_halo_mass")
        v = col("fof_halo_com_vx")
        # scalar arithmetic: m.mean() + v.std() is a DerivedScalarValue
        combined = m.mean() + v.std()
        assert combined.requires_names == {"fof_halo_mass", "fof_halo_com_vx"}

    def test_scalar_arithmetic_with_plain_number(self):
        m = col("fof_halo_mass")
        shifted = m.mean() + 1.0
        assert shifted.requires_names == {"fof_halo_mass"}

    def test_bound_scalar_same_result(self):
        m = col("fof_halo_mass")
        scalar = m.mean()
        scalar.name = "mean_mass"
        name_to_uuid = {"fof_halo_mass": uuid4()}
        bound = scalar.bind(name_to_uuid, name_to_uuid.values())
        assert bound.requires_names == {"fof_halo_mass"}

    def test_returns_fresh_set(self):
        scalar = col("fof_halo_mass").mean()
        s1 = scalar.requires_names
        s2 = scalar.requires_names
        assert s1 == s2
        s1.add("extra")
        assert "extra" not in scalar.requires_names

    def test_nested_column_expression_in_scalar(self):
        # (col("a") * col("b")).mean() — the scalar's lhs is a Column
        expr = (col("a") * col("b")).mean()
        assert expr.requires_names == {"a", "b"}


# ---------------------------------------------------------------------------
# EvaluatedColumn
# ---------------------------------------------------------------------------


def _make_evaluated_column(requires, produces=None):
    """Helper to build a minimal EvaluatedColumn for testing."""
    if produces is None:
        produces = {"out"}

    def func(**kwargs):
        return np.zeros(1)

    func.__name__ = "func"
    return EvaluatedColumn(
        func,
        requires,
        produces,
        "numpy",
        {name: None for name in produces},
        EvaluateStrategy.ROW_WISE,
    )


class TestEvaluatedColumnRequiresNames:
    def test_unbound_returns_declared_names(self):
        ec = _make_evaluated_column({"col_a", "col_b"})
        assert ec.requires_names == {"col_a", "col_b"}

    def test_bound_returns_same_names(self):
        ec = _make_evaluated_column({"col_a", "col_b"})
        name_to_uuid = {"col_a": uuid4(), "col_b": uuid4()}
        bound = ec.bind(name_to_uuid, name_to_uuid.values())
        assert bound.requires_names == {"col_a", "col_b"}

    def test_returns_fresh_set(self):
        ec = _make_evaluated_column({"col_a"})
        s = ec.requires_names
        s.add("extra")
        assert ec.requires_names == {"col_a"}

    def test_binding_does_not_alter_result(self):
        ec = _make_evaluated_column({"col_a", "col_b"})
        before = ec.requires_names
        name_to_uuid = {"col_a": uuid4(), "col_b": uuid4()}
        bound = ec.bind(name_to_uuid, name_to_uuid.values())
        after = bound.requires_names
        assert before == after
