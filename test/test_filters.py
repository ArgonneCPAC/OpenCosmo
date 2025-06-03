import astropy.cosmology.units as cu
import numpy as np
import pytest

import opencosmo as oc
from opencosmo import col


@pytest.fixture
def input_path(snapshot_path):
    return snapshot_path / "haloproperties.hdf5"


@pytest.fixture
def max_mass(input_path):
    ds = oc.open(input_path)
    sod_mass = ds.data["sod_halo_mass"]
    sod_mass_unit = sod_mass.unit
    return 0.95 * sod_mass.max() * sod_mass_unit


def test_filters(input_path):
    ds = oc.open(input_path)
    ds = ds.filter(col("sod_halo_mass") > 0)
    data = ds.data
    assert data["sod_halo_mass"].min() > 0


def test_multi_filters_single_column(input_path, max_mass):
    ds = oc.open(input_path)

    ds = ds.filter(col("sod_halo_mass") > 0, col("sod_halo_mass") < max_mass)
    data = ds.data
    assert data["sod_halo_mass"].min() > 0
    assert data["sod_halo_mass"].max() < max_mass.value


def test_multi_filters_multi_columns(input_path, max_mass):
    ds = oc.open(input_path)

    ds = ds.filter(
        col("sod_halo_mass") > 0,
        col("sod_halo_mass") < max_mass,
        col("sod_halo_cdelta") > 0,
        col("sod_halo_cdelta") < 20,
    )
    data = ds.data
    assert data["sod_halo_mass"].min() > 0
    assert data["sod_halo_mass"].max() < max_mass.value
    assert data["sod_halo_cdelta"].min() > 0
    assert data["sod_halo_cdelta"].max() < 20


def test_chained_filters(input_path, max_mass):
    ds = oc.open(input_path)
    ds = ds.filter(col("sod_halo_mass") > 0).filter(col("sod_halo_mass") < max_mass)
    ds = ds.filter(col("sod_halo_cdelta") > 0).filter(col("sod_halo_cdelta") < 20)
    data = ds.data
    assert data["sod_halo_mass"].min() > 0
    assert data["sod_halo_mass"].max() < max_mass.value
    assert data["sod_halo_cdelta"].min() > 0
    assert data["sod_halo_cdelta"].max() < 20


def test_take_filter(input_path, max_mass):
    ds = oc.open(input_path).take(10000)
    ds = ds.filter(col("sod_halo_mass") > 0, col("sod_halo_mass") < max_mass)
    assert len(ds.data) < 10000


def test_filter_unit_transformation(input_path, max_mass):
    ds = oc.open(input_path)
    ds = ds.filter(col("sod_halo_mass") > 0, col("sod_halo_mass") < max_mass)

    data = ds.data
    scalefree_data = ds.with_units("scalefree").data
    assert (
        data["sod_halo_mass"].unit == scalefree_data["sod_halo_mass"].unit * cu.littleh
    )

    littleh = ds.cosmology.h
    assert np.all(
        np.isclose(data["sod_halo_mass"] * littleh, scalefree_data["sod_halo_mass"])
    )


def test_filter_leaves_original_dataset_unchanged(input_path, max_mass):
    ds = oc.open(input_path)
    ds = ds.filter(col("sod_halo_mass") > 0, col("sod_halo_mass") < max_mass)
    data = ds.data
    data_len = len(data)
    original_len = len(ds._Dataset__handler)
    assert data_len < original_len


def test_equals_filter(input_path):
    ds = oc.open(input_path)
    data = ds.data
    sod_mass = data["sod_halo_mass"]
    equals_test_value = np.random.choice(sod_mass)
    ds = ds.filter(col("sod_halo_mass") == equals_test_value)
    data = ds.data
    assert len(data) > 0
    assert np.all(data["sod_halo_mass"] == equals_test_value)


def test_isin_filter(input_path):
    ds = oc.open(input_path)
    halo_tags = ds.select("fof_halo_tag").take(10, at="random").data
    ds = ds.filter(col("fof_halo_tag").isin(halo_tags))
    assert len(ds.data) == 10
    assert set(ds.data["fof_halo_tag"]) == set(halo_tags)


def test_notequals_filter(input_path):
    ds = oc.open(input_path)
    data = ds.data
    sod_mass = data["sod_halo_mass"]
    notequals_test_value = np.random.choice(sod_mass)
    ds = ds.filter(col("sod_halo_mass") != notequals_test_value)
    data = ds.data
    assert len(data) > 0
    assert np.all(data["sod_halo_mass"] != notequals_test_value)


def test_invalid_filter(input_path):
    ds = oc.open(input_path)
    data = ds.data
    sod_mass = data["sod_halo_mass"]
    sod_mass_max = sod_mass.max()
    ds = ds.filter(col("sod_halo_mass") > sod_mass_max + 1)
    assert len(ds) == 0


def test_filter_invalid_column(input_path):
    ds = oc.open(input_path)
    with pytest.raises(ValueError):
        ds = ds.filter(col("invalid_column") > 0)
