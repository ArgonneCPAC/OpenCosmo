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
