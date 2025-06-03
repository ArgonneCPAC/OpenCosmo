import random

import numpy as np
import pytest

import opencosmo as oc


@pytest.fixture
def halo_properties_path(snapshot_path):
    return snapshot_path / "haloproperties.hdf5"


def test_contains():
    reg1 = oc.Box((85, 85, 85), (115, 115, 115))
    reg2 = oc.Box((87.5, 87.5, 87.5), (92.5, 92.5, 92.5))
    reg3 = oc.Box((80, 80, 80), (100, 100, 100))
    assert reg1.contains(reg2)
    assert not reg1.contains(reg3)
    assert not reg2.contains(reg1)


def test_interesects():
    reg1 = oc.Box((90, 90, 90), (110, 110, 110))
    reg2 = oc.Box((80, 80, 80), (100, 100, 100))
    assert reg1.intersects(reg2)
    assert reg2.intersects(reg1)


def test_neither():
    reg1 = oc.Box((92.5, 92.5, 92.5), (107.5, 107.5, 107.5))
    reg2 = oc.Box((72.5, 72.5, 72.5), (77.5, 77.5, 77.5))
    assert not reg1.intersects(reg2)
    assert not reg2.intersects(reg1)
    assert not reg1.contains(reg2)
    assert not reg2.contains(reg1)


def test_box_query(halo_properties_path):
    ds = oc.open(halo_properties_path).with_units("scalefree")
    p1 = tuple(random.uniform(30, 40) for _ in range(3))
    p2 = tuple(random.uniform(40, 50) for _ in range(3))
    reg1 = oc.Box(p1, p2)

    original_data = ds.data
    ds = ds.bound(reg1)
    data = ds.data
    for i, dim in enumerate(["x", "y", "z"]):
        name = f"fof_halo_center_{dim}"
        original_col = original_data[name]
        max_ = p2[i]
        min_ = p1[i]
        mask = (original_col < max_) & (original_col > min_)
        original_data = original_data[mask]

        col = data[name]
        col_min = col.min()
        col_max = col.max()
        assert col_min >= min_ and np.isclose(col_min, min_, 0.1)
        assert col_max <= max_ and np.isclose(col_max, max_, 0.1)

    assert len(original_data) == len(data)


def test_box_query_physical(halo_properties_path):
    ds = oc.open(halo_properties_path).with_units("physical")

    p1 = tuple(random.uniform(30, 40) for _ in range(3))
    p2 = tuple(random.uniform(50, 60) for _ in range(3))
    reg1 = oc.Box(p1, p2)

    original_data = ds.data
    ds = ds.bound(reg1)

    data = ds.data
    for i, dim in enumerate(["x", "y", "z"]):
        col = data[f"fof_halo_center_{dim}"]
        original_col = original_data[f"fof_halo_center_{dim}"]
        col_min = col.min()
        col_max = col.max()
        min_ = p1[i]
        max_ = p2[i]
        original_data_mask = (original_col > min_) & (original_col < max_)
        original_data = original_data[original_data_mask]
        assert col_min >= min_
        assert col_max <= max_

    assert set(data["fof_halo_tag"]) == set(original_data["fof_halo_tag"])


def test_box_query_chain(halo_properties_path):
    ds = oc.open(halo_properties_path).with_units("scalefree")
    p11 = (25, 32.5, 40)
    p12 = (35, 47.5, 60)
    reg1 = oc.Box(p11, p12)

    p21 = (28.5, 37, 46)
    p22 = (33.5, 45, 56)
    reg2 = oc.Box(p21, p22)

    ds = ds.bound(reg1)
    ds = ds.bound(reg2)
    data = ds.data
    for i, dim in enumerate(["x", "y", "z"]):
        col = data[f"fof_halo_center_{dim}"]
        min_ = p21[i]
        max_ = p22[i]
        min = col.min()
        max = col.max()
        assert min >= min_ and np.isclose(min, min_, 0.1)
        assert max <= max_ and np.isclose(max, max_, 0.1)

        assert max <= max_ and np.isclose(max, max_, 0.1)


def test_write_tree(halo_properties_path, tmp_path):
    ds = oc.open(halo_properties_path).with_units("scalefree")
    p1 = (25, 32.5, 40)
    p2 = (35, 47.5, 60)
    reg1 = oc.Box(p1, p2)

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
    p11 = (25, 32.5, 40)
    p12 = (35, 47.5, 60)
    reg1 = oc.Box(p11, p12)

    p21 = (28.5, 37, 46)
    p22 = (33.5, 45, 56)
    reg2 = oc.Box(p21, p22)
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
    p1 = (25, 25, 25)
    p2 = (35, 35, 35)
    reg1 = oc.Box(p1, p2)

    p1 = (65, 65, 65)
    p2 = (75, 75, 75)

    reg2 = oc.Box(p1, p2)

    ds = ds.bound(reg1)
    with pytest.raises(ValueError):
        ds = ds.bound(reg2)


def test_box_query_chain_write_failure(halo_properties_path, tmp_path):
    ds = oc.open(halo_properties_path).with_units("scalefree")
    p1 = (25, 25, 25)
    p2 = (35, 35, 35)
    reg1 = oc.Box(p1, p2)

    p1 = (65, 65, 65)
    p2 = (75, 75, 75)

    reg2 = oc.Box(p1, p2)

    ds = ds.bound(reg1)
    output_path = tmp_path / "bound_dataset.hdf5"
    oc.write(output_path, ds)

    new_ds = oc.open(output_path)
    with pytest.raises(ValueError):
        new_ds.bound(reg2)
