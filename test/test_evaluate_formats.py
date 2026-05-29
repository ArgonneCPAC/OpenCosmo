import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import opencosmo as oc
from opencosmo.dataset import state as st


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


# ---------------------------------------------------------------------------
# StructureCollection paths
# ---------------------------------------------------------------------------


@pytest.fixture
def halo_paths(snapshot_path):
    files = ["haloproperties.hdf5", "haloparticles.hdf5"]
    return [snapshot_path / f for f in files]


def _mean_x(format):
    """Per-structure function: mean of dm_particles 'x' coord. Returns a scalar
    in the user's format. fof_halo_center_x is a scalar Quantity that has had
    its unit stripped for non-astropy formats, so it's plain float."""

    def offset(halo_properties, dm_particles):
        x = dm_particles["x"]
        if format == "arrow":
            mean_x = pc.mean(x).as_py()
        elif format == "polars":
            mean_x = x.mean()
        elif format == "pandas":
            mean_x = float(x.mean())
        elif format == "jax":
            mean_x = float(jnp.mean(x))
        else:
            mean_x = float(np.mean(x))
        return mean_x - float(halo_properties["fof_halo_center_x"])

    return offset


def _arange_like(format):
    """Per-structure function with dataset=`dm_particles`: must return an array
    in the user's format with the same length as the input dataset."""

    def particle_id(x, y, z):
        n = len(x)
        if format == "jax":
            return jnp.arange(n)
        if format == "pandas":
            return pd.Series(np.arange(n))
        if format == "polars":
            return pl.Series(values=np.arange(n))
        if format == "arrow":
            return pa.array(np.arange(n))
        return np.arange(n)

    return particle_id


@pytest.mark.parametrize("format", FORMATS)
def test_collection_evaluate_into_properties(halo_paths, format):
    collection = oc.open(*halo_paths).take(50)
    spec = {
        "dm_particles": ["x"],
        "halo_properties": ["fof_halo_center_x"],
    }
    collection = collection.evaluate(
        _mean_x(format), **spec, format=format, insert=True
    )
    data = collection["halo_properties"].select("offset").get_data("numpy")
    assert len(data) == 50
    assert np.any(data != 0)


@pytest.mark.parametrize("format", FORMATS)
def test_collection_evaluate_into_properties_noinsert(halo_paths, format):
    collection = oc.open(*halo_paths).take(50)
    spec = {
        "dm_particles": ["x"],
        "halo_properties": ["fof_halo_center_x"],
    }
    result = collection.evaluate(_mean_x(format), **spec, format=format, insert=False)
    assert "offset" in result
    assert len(result["offset"]) == 50


@pytest.mark.parametrize("format", FORMATS)
def test_collection_evaluate_into_dataset(halo_paths, format):
    collection = oc.open(*halo_paths).take(20)
    collection = collection.evaluate(
        _arange_like(format),
        dataset="dm_particles",
        format=format,
        insert=True,
    )
    for halo in collection.halos(["dm_particles"]):
        pid = halo["dm_particles"].select("particle_id").get_data("numpy")
        assert np.all(pid == np.arange(len(pid)))


@pytest.mark.parametrize("format", FORMATS)
def test_collection_evaluate_on_dataset(halo_paths, format):
    """Routes through Dataset.evaluate via the collection wrapper."""
    collection = oc.open(*halo_paths).take(50)
    selected = (
        collection["halo_properties"]
        .select(["fof_halo_mass", "fof_halo_com_vx"])
        .get_data("numpy")
    )
    collection = collection.evaluate_on_dataset(
        _vectorized_func(format),
        dataset="halo_properties",
        vectorize=True,
        format=format,
        insert=True,
    )
    data = collection["halo_properties"].select("fof_px").get_data("numpy")
    expected = selected["fof_halo_mass"] * selected["fof_halo_com_vx"]
    assert np.allclose(data, expected)


# ---------------------------------------------------------------------------
# Lightcone paths
# ---------------------------------------------------------------------------


@pytest.fixture
def lc_paths(lightcone_path):
    return [
        lightcone_path / "step_600" / "haloproperties.hdf5",
        lightcone_path / "step_601" / "haloproperties.hdf5",
    ]


@pytest.mark.parametrize("format", FORMATS)
def test_lightcone_evaluate_insert(lc_paths, format):
    ds = oc.open(*lc_paths).take(100)
    ds = ds.evaluate(
        _vectorized_func(format), vectorize=True, insert=True, format=format
    )
    for name in ds.keys():
        data = st.get_data(st.select(ds[name], {"fof_px"}), "numpy")
        original = st.get_data(
            st.select(ds[name], {"fof_halo_mass", "fof_halo_com_vx"}), "numpy"
        )
        expected = original["fof_halo_mass"] * original["fof_halo_com_vx"]
        assert np.allclose(data, expected)


@pytest.mark.parametrize("format", FORMATS)
def test_lightcone_evaluate_noinsert(lc_paths, format):
    ds = oc.open(*lc_paths).take(100)
    result = ds.evaluate(
        _vectorized_func(format), vectorize=True, insert=False, format=format
    )
    assert "fof_px" in result
    assert len(result["fof_px"]) == len(ds)
