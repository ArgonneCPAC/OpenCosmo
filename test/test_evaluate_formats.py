import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import opencosmo as oc


@pytest.fixture
def input_path(snapshot_path):
    return snapshot_path / "haloproperties.hdf5"


FORMATS = ["jax", "pandas", "polars", "arrow"]
SCALARS = {
    "jax": (jnp.ndarray, np.floating, float),
    "pandas": (pd.Series, np.floating, float),
    "polars": (pl.Series, float, int),
    "arrow": (pa.Scalar, float, int),
}


def _vectorized_func(format):
    """Multiply two columns using each format's native multiplication path."""

    if format == "arrow":

        def fof_px(fof_halo_mass, fof_halo_com_vx):
            return pc.multiply(fof_halo_mass, fof_halo_com_vx)

    else:

        def fof_px(fof_halo_mass, fof_halo_com_vx):
            return fof_halo_mass * fof_halo_com_vx

    return fof_px


def _row_func(format):
    """Multiply two scalars; works for any format because each row is a scalar."""

    def fof_px(fof_halo_mass, fof_halo_com_vx):
        if isinstance(fof_halo_mass, pa.Scalar):
            return fof_halo_mass.as_py() * fof_halo_com_vx.as_py()
        return float(fof_halo_mass) * float(fof_halo_com_vx)

    return fof_px


def _expected(input_path):
    data = (
        oc.open(input_path)
        .select(["fof_halo_mass", "fof_halo_com_vx"])
        .get_data("numpy")
    )
    return data["fof_halo_mass"] * data["fof_halo_com_vx"]


def _to_numpy(value):
    if isinstance(value, jnp.ndarray):
        return np.asarray(value)
    if isinstance(value, (pd.Series, pl.Series)):
        return value.to_numpy()
    if isinstance(value, pa.Array):
        return value.to_numpy(zero_copy_only=False)
    return np.asarray(value)


# ---------------------------------------------------------------------------
# insert = False (return result directly)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("format", FORMATS)
def test_evaluate_vectorized_noinsert(input_path, format):
    ds = oc.open(input_path)
    result = ds.evaluate(
        _vectorized_func(format), vectorize=True, insert=False, format=format
    )
    expected = _expected(input_path)
    assert np.allclose(_to_numpy(result["fof_px"]), expected)


@pytest.mark.parametrize("format", FORMATS)
def test_evaluate_row_wise_noinsert(input_path, format):
    ds = oc.open(input_path).take(500, at="start")
    result = ds.evaluate(
        _row_func(format), vectorize=False, insert=False, format=format
    )
    selected = (
        oc.open(input_path)
        .take(500, at="start")
        .select(["fof_halo_mass", "fof_halo_com_vx"])
        .get_data("numpy")
    )
    expected = selected["fof_halo_mass"] * selected["fof_halo_com_vx"]
    assert np.allclose(_to_numpy(result["fof_px"]), expected)


@pytest.mark.parametrize("format", FORMATS)
def test_evaluate_batched_noinsert(input_path, format):
    ds = oc.open(input_path)
    batch_size = 10_000
    result = ds.evaluate(
        _vectorized_func(format),
        insert=False,
        batch_size=batch_size,
        format=format,
    )
    expected = _expected(input_path)
    assert np.allclose(_to_numpy(result["fof_px"]), expected)


# ---------------------------------------------------------------------------
# insert = True (converted to numpy, stored in cache)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("format", FORMATS)
def test_evaluate_vectorized_insert(input_path, format):
    ds = oc.open(input_path)
    ds = ds.evaluate(
        _vectorized_func(format), vectorize=True, insert=True, format=format
    )
    assert "fof_px" in ds.columns
    data = ds.select("fof_px").get_data("numpy")
    expected = _expected(input_path)
    assert np.allclose(data, expected)


@pytest.mark.parametrize("format", FORMATS)
def test_evaluate_row_wise_insert(input_path, format):
    ds = oc.open(input_path).take(500, at="start")
    ds = ds.evaluate(_row_func(format), vectorize=False, insert=True, format=format)
    assert "fof_px" in ds.columns
    data = ds.select("fof_px").get_data("numpy")
    selected = (
        oc.open(input_path)
        .take(500, at="start")
        .select(["fof_halo_mass", "fof_halo_com_vx"])
        .get_data("numpy")
    )
    expected = selected["fof_halo_mass"] * selected["fof_halo_com_vx"]
    assert np.allclose(data, expected)


@pytest.mark.parametrize("format", FORMATS)
def test_evaluate_batched_insert(input_path, format):
    ds = oc.open(input_path)
    batch_size = 10_000
    ds = ds.evaluate(
        _vectorized_func(format),
        insert=True,
        batch_size=batch_size,
        format=format,
    )
    assert "fof_px" in ds.columns
    data = ds.select("fof_px").get_data("numpy")
    expected = _expected(input_path)
    assert np.allclose(data, expected)


# ---------------------------------------------------------------------------
# Output-type assertions on the not-insert path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "format,expected_type",
    [
        ("jax", jnp.ndarray),
        ("pandas", pd.Series),
        ("polars", pl.Series),
        ("arrow", pa.Array),
    ],
)
def test_evaluate_noinsert_returns_native_container(input_path, format, expected_type):
    ds = oc.open(input_path)
    result = ds.evaluate(
        _vectorized_func(format), vectorize=True, insert=False, format=format
    )
    assert isinstance(result["fof_px"], expected_type)
