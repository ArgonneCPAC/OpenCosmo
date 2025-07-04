import numpy as np
import pytest

import opencosmo as oc


@pytest.fixture
def input_path(snapshot_path):
    return snapshot_path / "galaxyproperties.hdf5"


def test_take_front(input_path):
    ds = oc.open(input_path)
    data = ds.data
    ds = ds.take(10, at="start")
    short_data = ds.data

    assert len(short_data) == 10
    cols = data.columns
    for col in cols:
        assert np.all(data[col][:10] == short_data[col])


def test_take_back(input_path):
    ds = oc.open(input_path)
    data = ds.data
    ds = ds.take(10, at="end")
    short_data = ds.data

    assert len(short_data) == 10
    cols = data.columns
    for col in cols:
        assert np.all(data[col][-10:] == short_data[col])


def test_take_random(input_path):
    ds = oc.open(input_path)
    front = ds.take(10).data
    end = ds.take(10, at="end").data
    short_data = ds.take(10, at="random").data

    assert not all(short_data == front)
    assert not all(short_data == end)


def test_take_chain(input_path):
    ds = oc.open(input_path)
    long_data = ds.data
    ds = ds.take(50, at="start")
    ds = ds.take(10, at="end")
    short_data = ds.data
    long_data = long_data[40:50]
    assert all(short_data == long_data)


def test_take_too_many(input_path):
    ds = oc.open(input_path)
    length = len(ds.data)
    with pytest.raises(ValueError):
        ds.take(length + 1)
