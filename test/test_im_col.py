import astropy.units as u
import numpy as np
import pytest

import opencosmo as oc


@pytest.fixture
def properties_path(snapshot_path):
    return snapshot_path / "haloproperties.hdf5"


def test_add_column(properties_path):
    ds = oc.open(properties_path)
    random_data = np.random.randint(0, 1000, size=len(ds))
    ds = ds.with_new_columns(test_random=random_data)
    assert "test_random" in ds.columns
    assert np.all(ds.select("test_random").get_data("numpy") == random_data)


def test_add_column_filter(properties_path):
    ds = oc.open(properties_path)
    random_data = np.random.randint(0, 1000, size=len(ds))
    ds = ds.with_new_columns(test_random=random_data)
    ds = ds.filter(oc.col("test_random") > 200, oc.col("test_random") < 500)
    assert np.all(ds.select("test_random").get_data("numpy") > 200)
    assert np.all(ds.select("test_random").get_data("numpy") < 500)


def test_add_quantity(properties_path):
    ds = oc.open(properties_path)
    random_data = np.random.randint(0, 1000, size=len(ds)) * u.deg
    ds = ds.with_new_columns(test_random=random_data)
    assert "test_random" in ds.columns
    assert np.all(ds.select("test_random").data == random_data)
    assert ds.select("test_random").data.unit == u.deg


def test_add_quantity_filter(properties_path):
    ds = oc.open(properties_path)
    random_data = np.random.randint(0, 1000, size=len(ds)) * u.deg
    ds = ds.with_new_columns(test_random=random_data)
    ds = ds.filter(oc.col("test_random") < 200)
    print(ds.select("test_random").get_data())
    assert np.all(ds.select("test_random").get_data() < 200 * u.deg)


def test_add_take(properties_path):
    ds = oc.open(properties_path)
    random_data = np.random.randint(0, 1000, size=len(ds))
    ds = ds.with_new_columns(test_random=random_data)
    ds = ds.take(100)
    index = ds.index.into_array()
    assert "test_random" in ds.columns
    assert np.all(ds.select("test_random").get_data("numpy") == random_data[index])


def test_add_derive(properties_path):
    ds = oc.open(properties_path)
    random_data = np.random.randint(0, 1000, size=len(ds))
    ds = ds.with_new_columns(test_random=random_data)
    ds = ds.with_new_columns(
        random_mass=oc.col("fof_halo_mass") * oc.col("test_random")
    )
    ds = ds.select(["fof_halo_mass", "test_random", "random_mass"])
    data = ds.get_data("numpy")
    assert np.all(data["random_mass"] == data["fof_halo_mass"] * data["test_random"])


def test_add_quantity_derive(properties_path):
    ds = oc.open(properties_path)
    random_data = np.random.randint(0, 1000, size=len(ds)) * (u.km / u.s)
    ds = ds.with_new_columns(test_random=random_data)
    ds = ds.with_new_columns(
        random_mass=oc.col("fof_halo_mass") * oc.col("test_random")
    )
    ds = ds.select(["fof_halo_mass", "test_random", "random_mass"])
    data = ds.get_data()
    assert np.all(data["random_mass"] == data["fof_halo_mass"] * data["test_random"])
    assert (
        data["random_mass"].unit
        == data["fof_halo_mass"].unit * data["test_random"].unit
    )


def test_add_derive_select(properties_path):
    ds = oc.open(properties_path)
    random_data = np.random.randint(0, 1000, size=len(ds))
    ds = ds.with_new_columns(test_random=random_data)
    ds = ds.with_new_columns(
        random_mass=oc.col("fof_halo_mass") * oc.col("test_random")
    )
    data_single = ds.select("random_mass").get_data("numpy")
    data_components = ds.select(["test_random", "fof_halo_mass"]).get_data("numpy")
    assert np.all(
        data_single == data_components["test_random"] * data_components["fof_halo_mass"]
    )


def test_add_derive_take(properties_path):
    ds = oc.open(properties_path)
    random_data = np.random.randint(0, 1000, size=len(ds))
    ds = ds.with_new_columns(test_random=random_data)
    ds = ds.with_new_columns(
        random_mass=oc.col("fof_halo_mass") * oc.col("test_random")
    )
    ds = ds.select(["fof_halo_mass", "test_random", "random_mass"]).take(100)
    data = ds.get_data("numpy")
    assert np.all(data["random_mass"] == data["fof_halo_mass"] * data["test_random"])


def test_add_derive_drop(properties_path):
    ds = oc.open(properties_path)
    random_data = np.random.randint(0, 1000, size=len(ds))
    ds = ds.with_new_columns(test_random=random_data)
    ds = ds.drop("test_random")
    data = ds.get_data("numpy")
    assert "test_random" not in data


def test_add_column_write(properties_path, tmp_path):
    ds = oc.open(properties_path)
    random_data = np.random.randint(0, 1000, size=len(ds))
    random_unitful = np.random.randint(0, 1000, size=len(ds)) * u.deg
    ds = ds.with_new_columns(test_random=random_data, test_unitful=random_unitful)
    oc.write(tmp_path / "test.hdf5", ds)
    ds = oc.open(tmp_path / "test.hdf5")
    assert "test_random" in ds.columns
    assert np.all(ds.select("test_random").get_data("numpy") == random_data)
    assert ds.select("test_random").get_data("numpy").dtype == random_data.dtype
    assert np.all(ds.select("test_unitful").get_data() == random_unitful)


def test_add_order(properties_path):
    ds = oc.open(properties_path)
    random_data = np.random.randint(0, 1000, size=len(ds))
    ds = ds.with_new_columns(test_random=random_data)
    ds = ds.sort_by("test_random")
    data = ds.get_data("numpy")
    test_random = data["test_random"]
    print(test_random)
    assert np.all(test_random[:-1] <= test_random[1:])
