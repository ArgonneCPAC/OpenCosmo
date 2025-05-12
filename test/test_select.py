import h5py
import numpy as np
import pytest
from astropy.cosmology import units as cu

from opencosmo import read


@pytest.fixture
def input_path(data_path):
    return data_path / "galaxyproperties.hdf5"


def test_select(input_path):
    dataset = read(input_path)
    data = dataset.data
    cols = list(data.columns)
    # select 10 columns at random
    selected_cols = np.random.choice(cols, 10, replace=False)
    selected = dataset.select(selected_cols)
    selected_data = selected.data

    for col in selected_cols:
        assert np.all(data[col] == selected_data[col])
    assert set(selected_cols) == set(selected_data.columns)


def test_chained_select(input_path):
    dataset = read(input_path)
    data = dataset.data
    cols = list(data.columns)
    # select 10 columns at random
    selected_cols = np.random.choice(cols, 10, replace=False)
    subset_cols = np.random.choice(selected_cols, 5, replace=False)
    selected = dataset.select(selected_cols).select(subset_cols)
    selected_data = selected.data

    for col in subset_cols:
        assert np.all(data[col] == selected_data[col])

    assert set(subset_cols) == set(selected_data.columns)


def test_select_unit_transformation(input_path):
    dataset = read(input_path)
    data = dataset.data
    cols = list(data.columns)
    # select 10 columns at random

    position_cols = list(
        filter(lambda col: col.split("_")[-1] in ["x", "y", "z"], cols)
    )
    selected = dataset.select(position_cols).with_units("scalefree")
    position_cols = filter(lambda col: "angmom" not in col, position_cols)

    selected_data = selected.data
    for col in position_cols:
        assert data[col].unit == selected_data[col].unit * cu.littleh


def test_select_derived_column():
    # todo!
    assert True


def test_select_doesnt_alter_raw(input_path):
    dataset = read(input_path)
    data = dataset.data
    cols = list(data.columns)
    # select 10 columns at random
    selected_cols = np.random.choice(cols, 10, replace=False)
    selected = dataset.select(selected_cols)
    selected_data = selected.data

    raw_data = dataset._Dataset__handler._DatasetHandler__group
    assert all(isinstance(raw_data[col], h5py.Dataset) for col in cols)
    assert all(data[col].unit == selected_data[col].unit for col in selected_cols)
    assert not all(np.all(data[col].value == raw_data[col][:]) for col in selected_cols)


def test_single_column_select(input_path):
    dataset = read(input_path)
    data = dataset.data
    cols = list(data.columns)
    # select 1 column at random
    selected_col = np.random.choice(cols, 1)[0]
    selected = dataset.select(selected_col)
    selected_data = selected.data

    assert np.all(data[selected_col] == selected_data)


def test_select_invalid_column(input_path):
    dataset = read(input_path)
    data = dataset.data
    cols = list(data.columns)
    # select 10 columns at random
    selected_cols = np.random.choice(cols, 10, replace=False)
    # add an invalid column
    selected_cols = np.append(selected_cols, "invalid_column")
    with pytest.raises(ValueError):
        dataset.select(selected_cols)
