import random

import numpy as np
import pytest

import opencosmo as oc


@pytest.fixture
def halo_properties_path(snapshot_path):
    return snapshot_path / "haloproperties.hdf5"


def test_contains():
    reg1 = oc.Box((100, 100, 100), 30)
    reg2 = oc.Box((90, 90, 90), 5)
    reg3 = oc.Box((90, 90, 90), 20)
    assert reg1.contains(reg2)
    assert not reg1.contains(reg3)
    assert not reg2.contains(reg1)


def test_interesects():
    reg1 = oc.Box((100, 100, 100), 20)
    reg2 = oc.Box((90, 90, 90), 20)
    assert reg1.intersects(reg2)
    assert reg2.intersects(reg1)


def test_neither():
    reg1 = oc.Box((100, 100, 100), 15)
    reg2 = oc.Box((75, 75, 75), 5)
    assert not reg1.intersects(reg2)
    assert not reg2.intersects(reg1)
    assert not reg1.contains(reg2)
    assert not reg2.contains(reg1)


def test_box_query(halo_properties_path):
    ds = oc.open(halo_properties_path).with_units("scalefree")
    center = tuple(random.uniform(30, 60) for _ in range(3))
    width = tuple(random.uniform(10, 20) for _ in range(3))
    reg1 = oc.Box(center, width)
    original_data = ds.data
    ds = ds.bound(reg1)
    data = ds.data
    for i, dim in enumerate(["x", "y", "z"]):
        name = f"fof_halo_center_{dim}"
        min_ = center[i] - width[i] / 2
        max_ = center[i] + width[i] / 2
        original_col = original_data[name]
        mask = (original_col < max_) & (original_col > min_)
        original_data = original_data[mask]

        col = data[name]
        min = col.min()
        max = col.max()
        assert min >= min_ and np.isclose(min, min_, 0.1)
        assert max <= max_ and np.isclose(max, max_, 0.1)

    assert len(original_data) == len(data)


def test_box_query_physical(halo_properties_path):
    ds = oc.open(halo_properties_path).with_units("physical")
    center = tuple(random.uniform(30, 60) for _ in range(3))
    width = tuple(random.uniform(10, 20) for _ in range(3))
    reg1 = oc.Box(center, width)
    ds = ds.bound(reg1)
    data = ds.data
    for i, dim in enumerate(["x", "y", "z"]):
        col = data[f"fof_halo_center_{dim}"]
        min = col.min()
        max = col.max()
        min_ = center[i] - width[i] / 2
        max_ = center[i] + width[i] / 2
        assert min >= min_ and np.isclose(min, min_, 0.1)
        assert max <= max_ and np.isclose(max, max_, 0.1)


def test_box_query_chain(halo_properties_path):
    ds = oc.open(halo_properties_path).with_units("scalefree")
    center1 = (30, 40, 50)
    width1 = (10, 15, 20)
    reg1 = oc.Box(center1, width1)

    center2 = (31, 41, 51)
    width2 = (5, 7.5, 10)
    reg2 = oc.Box(center2, width2)

    ds = ds.bound(reg1)
    ds = ds.bound(reg2)
    data = ds.data
    for i, dim in enumerate(["x", "y", "z"]):
        col = data[f"fof_halo_center_{dim}"]
        min_ = center2[i] - width2[i] / 2
        max_ = center2[i] + width2[i] / 2
        min = col.min()
        max = col.max()
        assert min >= min_ and np.isclose(min, min_, 0.1)
        assert max <= max_ and np.isclose(max, max_, 0.1)

        assert max <= max_ and np.isclose(max, max_, 0.1)


def test_write_tree(halo_properties_path, tmp_path):
    ds = oc.open(halo_properties_path).with_units("scalefree")
    center1 = (30, 40, 50)
    width1 = (10, 15, 20)
    reg1 = oc.Box(center1, width1)

    ds = ds.bound(reg1)
    oc.write(tmp_path / "bound_dataset.hdf5", ds)

    ds = oc.open(tmp_path / "bound_dataset.hdf5").with_units("scalefree")
    tree_data = ds._Dataset__tree._Tree__data
    for i in range(3):
        level = tree_data[f"level_{i}"]
        starts = level["start"][:]
        sizes = level["size"][:]
        assert np.sum(sizes) == len(ds)
        assert np.all(np.insert(np.cumsum(sizes), 0, 0)[:-1] == starts)


def test_box_query_chain_with_write(halo_properties_path, tmp_path):
    ds = oc.open(halo_properties_path).with_units("scalefree")
    center1 = (30, 40, 50)
    width1 = (10, 15, 20)
    reg1 = oc.Box(center1, width1)

    center2 = (31, 41, 51)
    width2 = (5, 7.5, 10)
    reg2 = oc.Box(center2, width2)

    ds = ds.bound(reg1)
    oc.write(tmp_path / "bound_dataset.hdf5", ds)

    ds2 = oc.open(tmp_path / "bound_dataset.hdf5").with_units("scalefree")

    ds = ds.bound(reg2)
    ds2 = ds2.bound(reg2)

    data = ds.data
    data2 = ds2.data

    for i, dim in enumerate(["x", "y", "z"]):
        col = data[f"fof_halo_center_{dim}"]
        col2 = data2[f"fof_halo_center_{dim}"]
        assert np.all(col == col2)


def test_box_query_chain_failure(halo_properties_path):
    ds = oc.open(halo_properties_path).with_units("scalefree")
    center1 = (30, 30, 30)
    width1 = (10, 10, 10)
    reg1 = oc.Box(center1, width1)

    center2 = (70, 70, 70)
    width2 = (10, 10, 10)
    reg2 = oc.Box(center2, width2)

    ds = ds.bound(reg1)
    with pytest.raises(ValueError):
        ds = ds.bound(reg2)
