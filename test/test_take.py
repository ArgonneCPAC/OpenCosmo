import numpy as np
import pytest

import opencosmo as oc


@pytest.fixture
def input_path(snapshot_path):
    return snapshot_path / "galaxyproperties.hdf5"


def test_take_front(input_path):
    ds = oc.open(input_path)
    data = ds.get_data()
    ds = ds.take(10, at="start")
    short_data = ds.get_data()

    assert len(short_data) == 10
    cols = data.columns
    for col in cols:
        assert np.all(data[col][:10] == short_data[col])


def test_take_back(input_path):
    ds = oc.open(input_path)
    data = ds.get_data()
    ds = ds.take(10, at="end")
    short_data = ds.get_data()

    assert len(short_data) == 10
    cols = data.columns
    for col in cols:
        assert np.all(data[col][-10:] == short_data[col])


def test_take_random(input_path):
    ds = oc.open(input_path)
    front = ds.take(10).get_data()
    end = ds.take(10, at="end").get_data()
    short_data = ds.take(10, at="random").get_data()

    assert not all(short_data == front)
    assert not all(short_data == end)


def test_take_chain(input_path):
    ds = oc.open(input_path)
    long_data = ds.get_data()
    ds = ds.take(50, at="start")
    ds = ds.take(10, at="end")
    short_data = ds.get_data()
    long_data = long_data[40:50]

    for column in short_data.columns:
        assert np.all(short_data[column] == long_data[column])


def test_take_too_many(input_path):
    ds = oc.open(input_path)
    length = len(ds.get_data())

    new_ds = ds.take(length + 1)
    assert len(new_ds) == len(ds)


def test_take_end_too_many(input_path):
    ds = oc.open(input_path)
    length = len(ds)

    new_ds = ds.take(length + 1, at="end")
    assert len(new_ds) == length


def test_take_end_sorted(input_path):
    ds = oc.open(input_path)
    cols = ds.columns
    sort_col = cols[0]
    n = 10

    all_values = ds.select(sort_col).get_data("numpy")
    threshold = np.sort(all_values)[-n]

    taken = ds.sort_by(sort_col).take(n, at="end").select(sort_col).get_data("numpy")

    assert len(taken) == n
    assert np.all(taken >= threshold)
