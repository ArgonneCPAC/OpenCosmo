import numpy as np
import pytest
from opencosmo.header import read_header, write_header

import opencosmo as oc
from opencosmo import col, write


@pytest.fixture
def cosmology_resource_path(test_data):
    return test_data.snapshot.header


@pytest.fixture
def halo_properties_path(test_data):
    return test_data.snapshot.primary.halo_properties


@pytest.fixture
def galaxy_properties_path(test_data):
    return test_data.snapshot.primary.galaxy_properties


def test_write_header(test_data, tmp_path):
    header = read_header(test_data.snapshot.primary.galaxy_properties)
    new_path = tmp_path / "header.hdf5"
    write_header(new_path, header)

    new_header = read_header(new_path)
    assert header.simulation == new_header.simulation


def test_write_dataset(halo_properties_path, tmp_path):
    ds = oc.open(halo_properties_path)
    new_path = tmp_path / "haloproperties.hdf5"
    write(new_path, ds)

    new_ds = oc.open(new_path)
    assert all(ds.get_data() == new_ds.get_data())


def test_overwrite(halo_properties_path, tmp_path):
    ds = oc.open(halo_properties_path)
    new_path = tmp_path / "haloproperties.hdf5"
    write(new_path, ds)
    with pytest.raises(FileExistsError):
        write(new_path, ds)
    write(new_path, ds, overwrite=True)

    new_ds = oc.open(new_path)
    assert all(ds.get_data() == new_ds.get_data())


def test_after_take_filter(halo_properties_path, tmp_path):
    ds = oc.open(halo_properties_path).take(10000)
    ds = ds.filter(col("sod_halo_mass") > 0)
    filtered_data = ds.get_data()

    write(tmp_path / "haloproperties.hdf5", ds)
    new_ds = oc.open(tmp_path / "haloproperties.hdf5")
    assert np.all(filtered_data == new_ds.get_data())


def test_after_take(halo_properties_path, tmp_path):
    ds = oc.open(halo_properties_path).take(10000)
    data = ds.get_data()
    write(tmp_path / "haloproperties.hdf5", ds)

    new_ds = oc.open(tmp_path / "haloproperties.hdf5")
    assert all(data == new_ds.get_data())


def test_after_filter(halo_properties_path, tmp_path):
    ds = oc.open(halo_properties_path)
    data = ds.get_data()
    ds = ds.filter(col("sod_halo_mass") > 0)
    filtered_data = ds.get_data()
    assert len(data) > len(filtered_data)

    write(tmp_path / "haloproperties.hdf5", ds)

    new_ds = oc.open(tmp_path / "haloproperties.hdf5")
    assert all(filtered_data == new_ds.get_data())


def test_after_unit_transform(halo_properties_path, tmp_path):
    ds = oc.open(halo_properties_path)
    ds = ds.with_units("scalefree")

    # write should not change the data
    write(tmp_path / "haloproperties.hdf5", ds)

    ds = oc.open(halo_properties_path)
    new_ds = oc.open(tmp_path / "haloproperties.hdf5")
    assert all(ds.get_data() == new_ds.get_data())
