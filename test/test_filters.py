import astropy.cosmology.units as cu
import astropy.units as u
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
    sod_mass = ds.get_data()["sod_halo_mass"]
    return 0.95 * sod_mass.max()


def test_filters(input_path):
    ds = oc.open(input_path)
    ds = ds.filter(col("sod_halo_mass") > 0)
    data = ds.get_data()
    assert data["sod_halo_mass"].min() > 0


def test_multi_filters_single_column(input_path, max_mass):
    ds = oc.open(input_path)

    ds = ds.filter(col("sod_halo_mass") > 0, col("sod_halo_mass") < max_mass)
    data = ds.get_data()
    assert data["sod_halo_mass"].min() > 0
    assert data["sod_halo_mass"].max() < max_mass


def test_multi_filters_multi_columns(input_path, max_mass):
    ds = oc.open(input_path)

    ds = ds.filter(
        col("sod_halo_mass") > 0,
        col("sod_halo_mass") < max_mass,
        col("sod_halo_cdelta") > 0,
        col("sod_halo_cdelta") < 20,
    )
    data = ds.get_data()
    assert data["sod_halo_mass"].min().value > 0
    assert data["sod_halo_mass"].max() < max_mass
    assert data["sod_halo_cdelta"].min() > 0
    assert data["sod_halo_cdelta"].max() < 20


def test_chained_filters(input_path, max_mass):
    ds = oc.open(input_path)
    ds = ds.filter(col("sod_halo_mass") > 0).filter(col("sod_halo_mass") < max_mass)
    ds = ds.filter(col("sod_halo_cdelta") > 0).filter(col("sod_halo_cdelta") < 20)
    data = ds.get_data()
    assert data["sod_halo_mass"].min() > 0
    assert data["sod_halo_mass"].max().value < max_mass.value
    assert data["sod_halo_cdelta"].min() > 0
    assert data["sod_halo_cdelta"].max() < 20


def test_take_filter(input_path, max_mass):
    ds = oc.open(input_path).take(10000)
    ds = ds.filter(col("sod_halo_mass") > 0, col("sod_halo_mass") < max_mass)
    assert len(ds.get_data()) < 10000


def test_filter_unit_transformation(input_path, max_mass):
    ds = oc.open(input_path)
    ds = ds.filter(col("sod_halo_mass") > 0, col("sod_halo_mass") < max_mass)

    data = ds.get_data()
    scalefree_data = ds.with_units("scalefree").get_data()
    assert (
        data["sod_halo_mass"].unit == scalefree_data["sod_halo_mass"].unit * cu.littleh
    )

    littleh = ds.cosmology.h
    assert np.all(
        np.isclose(
            data["sod_halo_mass"].value * littleh, scalefree_data["sod_halo_mass"].value
        )
    )


def test_filter_leaves_original_dataset_unchanged(input_path, max_mass):
    ds = oc.open(input_path)
    ds = ds.filter(col("sod_halo_mass") > 0, col("sod_halo_mass") < max_mass)
    data = ds.get_data()
    data_len = len(data)
    original_len = len(ds._Dataset__state._DatasetState__raw_data_handler)
    assert data_len < original_len


def test_equals_filter(input_path):
    ds = oc.open(input_path)
    data = ds.get_data()
    sod_mass = data["sod_halo_mass"]
    equals_test_value = np.random.choice(sod_mass)
    ds = ds.filter(col("sod_halo_mass") == equals_test_value)
    data = ds.get_data()
    assert len(data) > 0
    assert np.all(data["sod_halo_mass"] == equals_test_value * u.Msun)


def test_isin_filter(input_path):
    ds = oc.open(input_path)
    halo_tags = ds.select("fof_halo_tag").take(10, at="random").get_data()
    ds = ds.filter(col("fof_halo_tag").isin(halo_tags))
    assert len(ds.get_data()) == 10
    assert set(ds.get_data()["fof_halo_tag"]) == set(halo_tags)


def test_notequals_filter(input_path):
    ds = oc.open(input_path)
    data = ds.get_data()
    sod_mass = data["sod_halo_mass"]
    notequals_test_value = np.random.choice(sod_mass)
    ds = ds.filter(col("sod_halo_mass") != notequals_test_value)
    data = ds.get_data()
    assert len(data) > 0
    assert np.all(data["sod_halo_mass"] != notequals_test_value)


def test_invalid_filter(input_path):
    ds = oc.open(input_path)
    data = ds.get_data()
    sod_mass = data["sod_halo_mass"]
    sod_mass_max = sod_mass.max()
    ds = ds.filter(col("sod_halo_mass") > sod_mass_max.value + 1)
    assert len(ds) == 0


def test_filter_invalid_column(input_path):
    ds = oc.open(input_path)
    with pytest.raises(ValueError):
        ds = ds.filter(col("invalid_column") > 0)


def test_or_filter(input_path):
    ds = oc.open(input_path)
    high_mass = oc.col("fof_halo_mass") > 1e14
    low_mass = oc.col("fof_halo_mass") < 1e12

    ds = ds.filter(high_mass | low_mass)
    data = ds.select("fof_halo_mass").get_data("numpy")
    assert len(data) > 0
    assert np.all((data > 1e14) | (data < 1e12))
    assert np.any(data > 1e14)
    assert np.any(data < 1e12)


def test_and_filter(input_path):
    ds = oc.open(input_path)
    high_mass = oc.col("fof_halo_mass") > 1e14
    low_c = oc.col("sod_halo_cdelta") < 5

    ds = ds.filter(high_mass & low_c)
    data = ds.select(("fof_halo_mass", "sod_halo_cdelta")).get_data("numpy")
    assert len(data) > 0
    assert np.all((data["fof_halo_mass"] > 1e14) & (data["sod_halo_cdelta"] < 5))


def test_filter_tree(input_path):
    ds = oc.open(input_path)
    high_mass = oc.col("fof_halo_mass") > 1e14
    low_mass = oc.col("fof_halo_mass") < 1e12
    low_c = oc.col("sod_halo_cdelta") < 5
    mass_cuts = high_mass | low_mass

    ds = ds.filter(mass_cuts & low_c)
    data = ds.select(("fof_halo_mass", "sod_halo_cdelta")).get_data("numpy")
    assert len(data) > 0
    assert np.all((data["fof_halo_mass"] > 1e14) | (data["fof_halo_mass"] < 1e12))
    assert np.all(data["sod_halo_cdelta"] < 5)
    assert np.any(data["fof_halo_mass"] > 1e14)
    assert np.any(data["fof_halo_mass"] < 1e12)
