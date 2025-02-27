from pathlib import Path

import numpy as np
import pytest

from opencosmo import read


@pytest.fixture
def data_path():
    return Path("test/resource/galaxyproperties.hdf5")


def test_select(data_path):
    dataset = read(data_path)
    data = dataset.data
    cols = list(data.columns)
    # select 10 columns at random
    selected_cols = np.random.choice(cols, 10, replace=False)
    selected = dataset.select(selected_cols)
    selected_data = selected.data

    for col in selected_cols:
        assert np.all(data[col] == selected_data[col])


def test_select_derived_column():
    assert False


def test_select_doesnt_alter_raw(data_path):
    dataset = read(data_path)
    data = dataset.data
    cols = list(data.columns)
    # select 10 columns at random
    selected_cols = np.random.choice(cols, 10, replace=False)
    selected = dataset.select(selected_cols)
    selected_data = selected.data

    raw_data = dataset._OpenCosmoDataset__handler._InMemoryHandler__data
    assert all(raw_data[col].unit is None for col in selected_cols)
    assert all(data[col].unit == selected_data[col].unit for col in selected_cols)


def test_single_column_select(data_path):
    dataset = read(data_path)
    data = dataset.data
    cols = list(data.columns)
    # select 1 column at random
    selected_col = np.random.choice(cols, 1)[0]
    selected = dataset.select(selected_col)
    selected_data = selected.data

    assert np.all(data[selected_col] == selected_data[selected_col])
