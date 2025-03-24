import pytest

from opencosmo import col, read,  write
from opencosmo.header import read_header, write_header


@pytest.fixture
def cosmology_resource_path(data_path):
    p = data_path / "header.hdf5"
    return p


@pytest.fixture
def properties_path(data_path):
    return data_path / "haloproperties.hdf5"


def test_write_header(data_path, tmp_path):
    header = read_header(data_path / "galaxyproperties.hdf5")
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


def test_write_dataset(properties_path, tmp_path):
    ds = read(properties_path)
    new_path = tmp_path / "haloproperties.hdf5"
    write(new_path, ds)

    new_ds = read(new_path)
    assert all(ds.data == new_ds.data)


def test_after_filter(properties_path, tmp_path):
    ds = read(properties_path)
    data = ds.data
    ds = ds.filter(col("sod_halo_mass") > 0)
    filtered_data = ds.data
    assert len(data) > len(filtered_data)

    write(tmp_path / "haloproperties.hdf5", ds)

    new_ds = read(tmp_path / "haloproperties.hdf5")
    assert all(filtered_data == new_ds.data)


def test_after_unit_transform(properties_path, tmp_path):
    ds = read(properties_path)
    ds = ds.with_units("scalefree")

    # write should not change the data
    write(tmp_path / "haloproperties.hdf5", ds)

    ds = read(properties_path)
    new_ds = read(tmp_path / "haloproperties.hdf5")
    assert all(ds.data == new_ds.data)
