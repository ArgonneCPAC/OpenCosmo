import pytest

from opencosmo import col, read, write
from opencosmo.header import read_header, write_header


@pytest.fixture
def cosmology_resource_path(snapshot_path):
    p = snapshot_path / "header.hdf5"
    return p


@pytest.fixture
def halo_properties_path(snapshot_path):
    return snapshot_path / "haloproperties.hdf5"


@pytest.fixture
def galaxy_properties_path(snapshot_path):
    return snapshot_path / "galaxyproperties.hdf5"


def test_write_header(snapshot_path, tmp_path):
    header = read_header(snapshot_path / "galaxyproperties.hdf5")
    new_path = tmp_path / "header.hdf5"
    write_header(new_path, header)

    new_header = read_header(new_path)
    assert header.simulation == new_header.simulation
    assert (
        header._OpenCosmoHeader__reformat_pars
        == new_header._OpenCosmoHeader__reformat_pars
    )
    assert (
        header._OpenCosmoHeader__cosmotools_pars
        == new_header._OpenCosmoHeader__cosmotools_pars
    )


def test_write_dataset(halo_properties_path, tmp_path):
    ds = read(halo_properties_path)
    new_path = tmp_path / "haloproperties.hdf5"
    write(new_path, ds)

    new_ds = read(new_path)
    assert all(ds.data == new_ds.data)


def test_after_take_filter(halo_properties_path, tmp_path):
    ds = read(halo_properties_path).take(10000)
    ds = ds.filter(col("sod_halo_mass") > 0)
    filtered_data = ds.data

    write(tmp_path / "haloproperties.hdf5", ds)
    new_ds = read(tmp_path / "haloproperties.hdf5")
    assert all(filtered_data == new_ds.data)


def test_after_take(halo_properties_path, tmp_path):
    ds = read(halo_properties_path).take(10000)
    data = ds.data
    write(tmp_path / "haloproperties.hdf5", ds)

    new_ds = read(tmp_path / "haloproperties.hdf5")
    assert all(data == new_ds.data)


def test_after_filter(halo_properties_path, tmp_path):
    ds = read(halo_properties_path)
    data = ds.data
    ds = ds.filter(col("sod_halo_mass") > 0)
    filtered_data = ds.data
    assert len(data) > len(filtered_data)

    write(tmp_path / "haloproperties.hdf5", ds)

    new_ds = read(tmp_path / "haloproperties.hdf5")
    assert all(filtered_data == new_ds.data)


def test_after_unit_transform(halo_properties_path, tmp_path):
    ds = read(halo_properties_path)
    ds = ds.with_units("scalefree")

    # write should not change the data
    write(tmp_path / "haloproperties.hdf5", ds)

    ds = read(halo_properties_path)
    new_ds = read(tmp_path / "haloproperties.hdf5")
    assert all(ds.data == new_ds.data)
