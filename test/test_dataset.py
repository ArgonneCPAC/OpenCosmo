import numpy as np
import pytest
from astropy.table import Column

import opencosmo as oc


@pytest.fixture
def input_path(snapshot_path):
    return snapshot_path / "haloproperties.hdf5"


@pytest.fixture
def particle_path(snapshot_path):
    return snapshot_path / "haloparticles.hdf5"


@pytest.fixture
def max_mass(input_path):
    ds = oc.open(input_path)
    sod_mass = ds.data["sod_halo_mass"]
    sod_mass_unit = sod_mass.unit
    return 0.95 * sod_mass.max() * sod_mass_unit


def test_open(input_path):
    read_data = oc.open(input_path).data
    with oc.open(input_path) as f:
        open_data = f.data

    assert np.all(read_data == open_data)
    columns = read_data.columns
    assert all(open_data[col].unit == read_data[col].unit for col in columns)


def test_open_close(input_path):
    with oc.open(input_path) as ds:
        file = ds._Dataset__handler._DatasetHandler__file
        assert file["data"] is not None

    with pytest.raises(KeyError):
        file["data"]


def test_dataset_close(input_path):
    ds = oc.open(input_path)
    file = ds._Dataset__handler._DatasetHandler__file
    assert file["data"] is not None
    ds.close()
    with pytest.raises(KeyError):
        file["data"]


def test_filter_oom(input_path, max_mass):
    # Assuming test_open worked, this is the only
    # thing that needs to be directly tested

    with oc.open(input_path) as f:
        ds = f.filter(oc.col("sod_halo_mass") > 0, oc.col("sod_halo_mass") < max_mass)
        data = ds.data
    assert data["sod_halo_mass"].min() > 0


def test_filter_to_numpy(input_path, max_mass):
    # Assuming test_open worked, this is the only
    # thing that needs to be directly tested

    with oc.open(input_path) as f:
        ds = f.filter(oc.col("sod_halo_mass") > 0, oc.col("sod_halo_mass") < max_mass)
        data = ds.get_data(output="numpy")
    assert isinstance(data, dict)
    for val in data.values():
        assert isinstance(val, np.ndarray)
    assert data["sod_halo_mass"].min() > 0


def test_take_oom(input_path):
    with oc.open(input_path) as f:
        ds = f.take(10)
        data = ds.data
    assert len(data) == 10


def test_drop(input_path):
    with oc.open(input_path) as ds:
        data = ds.data
        cols = list(data.columns)
        # select 10 columns at random
        dropped_cols = np.random.choice(cols, 10, replace=False)
        selected = ds.drop(dropped_cols)
        selected_data = selected.data

    dropped_cols = set(dropped_cols)
    remaining_cols = set(selected_data.colnames)
    assert not dropped_cols.intersection(remaining_cols)


def test_drop_single(input_path):
    with oc.open(input_path) as ds:
        data = ds.data
        cols = list(data.columns)
        # select 10 columns at random
        dropped_col = np.random.choice(cols)
        remaining = ds.drop(dropped_col)
        remaining_data = remaining.data

    assert dropped_col not in remaining_data.colnames


def test_select_oom(input_path):
    with oc.open(input_path) as ds:
        data = ds.data
        cols = list(data.columns)
        # select 10 columns at random
        selected_cols = np.random.choice(cols, 10, replace=False)
        selected = ds.select(selected_cols)
        selected_data = selected.data

    for col in selected_cols:
        assert np.all(data[col] == selected_data[col])
    assert set(selected_cols) == set(selected_data.columns)


def test_select_single(input_path):
    with oc.open(input_path) as ds:
        data = ds.data
        cols = list(data.columns)
        # select 10 columns at random
        selected_cols = np.random.choice(cols)
        selected = ds.select(selected_cols)
        selected_data = selected.data
    assert isinstance(selected_data, Column)


def test_select_single_numpy(input_path):
    with oc.open(input_path) as ds:
        data = ds.data
        cols = list(data.columns)
        # select 10 columns at random
        selected_cols = np.random.choice(cols)
        selected = ds.select(selected_cols)
        selected_data = selected.get_data("numpy")
    assert isinstance(selected_data, np.ndarray)


def test_write_after_filter(input_path, tmp_path):
    with oc.open(input_path) as ds:
        ds = ds.filter(oc.col("sod_halo_mass") > 0)

        oc.write(tmp_path / "haloproperties.hdf5", ds)

        data = ds.data

    with oc.open(tmp_path / "haloproperties.hdf5") as new_ds:
        filtered_data = new_ds.data
        assert np.all(data == filtered_data)


def test_collect(input_path):
    with oc.open(input_path) as f:
        ds = f.filter(oc.col("sod_halo_mass") > 0).take(100, at="random").collect()

    assert len(ds.data) == 100


def test_select_collect(input_path):
    with oc.open(input_path) as f:
        ds = (
            f.filter(oc.col("sod_halo_mass") > 0)
            .select(["sod_halo_mass", "fof_halo_mass"])
            .take(100, at="random")
            .collect()
        )

    assert len(ds.data) == 100
    assert set(ds.data.columns) == {"sod_halo_mass", "fof_halo_mass"}
