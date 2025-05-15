import pytest

import opencosmo as oc
from opencosmo.spatial.region import BoxRegion


@pytest.fixture
def halo_properties_path(data_path):
    return data_path / "haloproperties.hdf5"


def test_contains():
    reg1 = BoxRegion((100, 100, 100), 15)
    reg2 = BoxRegion((90, 90, 90), 2.5)
    reg3 = BoxRegion((90, 90, 90), 10)
    assert reg1.contains(reg2)
    assert not reg1.contains(reg3)
    assert not reg2.contains(reg1)


def test_interesects():
    reg1 = BoxRegion((100, 100, 100), 20)
    reg2 = BoxRegion((90, 90, 90), 20)
    assert reg1.intersects(reg2)
    assert reg2.intersects(reg1)


def test_neither():
    reg1 = BoxRegion((100, 100, 100), 15)
    reg2 = BoxRegion((75, 75, 75), 5)
    assert not reg1.intersects(reg2)
    assert not reg2.intersects(reg1)
    assert not reg1.contains(reg2)
    assert not reg2.contains(reg1)


def test_box_query(halo_properties_path):
    ds = oc.open(halo_properties_path).with_units("scalefree")
    reg1 = BoxRegion((100, 100, 100), 15)
    ds = ds.crop(reg1)
    data = ds.data
    for dim in ["x", "y", "z"]:
        col = data[f"fof_halo_center_{dim}"]
        min = col.min()
        max = col.max()
        assert min >= 85 and min <= 85.1
        assert max <= 115 and max >= 114.9


def test_box_query_physical(halo_properties_path):
    ds = oc.open(halo_properties_path).with_units("physical")
    reg1 = BoxRegion((100, 100, 100), 15)
    ds = ds.crop(reg1)
    data = ds.data
    for dim in ["x", "y", "z"]:
        col = data[f"fof_halo_center_{dim}"]
        min = col.min()
        max = col.max()
        assert min >= 85 and min <= 85.1
        assert max <= 115 and max >= 114.9
