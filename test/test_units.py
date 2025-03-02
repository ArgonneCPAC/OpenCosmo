from pathlib import Path
from shutil import copyfile

import astropy.cosmology.units as cu
import astropy.units as u
import h5py
import numpy as np
import pytest
from astropy.table import Column

from opencosmo import read


def add_column(tmp_path: Path, original_file: Path, column: Column):
    new_path = tmp_path / original_file.name
    copyfile(original_file, new_path)
    with h5py.File(new_path, "r+") as file:
        data = file["data"]
        data.create_dataset(column.name, data=column.data)
        if column.unit is not None:
            data[column.name].attrs["unit"] = str(column.unit)
    return new_path


@pytest.fixture
def data_path():
    return Path("test/resource/galaxyproperties.hdf5")


def test_attribute_units(data_path, tmp_path):
    dataset = read(data_path)
    data = dataset.data
    new_column = Column(np.random.rand(len(data)), name="new_column", unit=u.Mpc)
    new_path = add_column(tmp_path, data_path, new_column)
    dataset = read(new_path)
    data = dataset.data
    assert data["new_column"].unit == u.Mpc


def test_parse_velocities(data_path):
    dataset = read(data_path)
    data = dataset.data
    cols = data.columns
    velocity_cols = filter(lambda col: col.upper()[-2:] in ["VX", "VY", "VZ"], cols)
    for col in velocity_cols:
        assert data[col].unit == u.km / u.s


def test_comoving_vs_scalefree(data_path):
    comoving = read(data_path, units="comoving")
    scalefree = read(data_path, units="scalefree")
    cols = comoving.data.columns
    position_cols = filter(lambda col: col.split("_")[-1] in ["x", "y", "z"], cols)
    h = comoving.cosmology.h
    for col in position_cols:
        assert scalefree.data[col].unit == u.Mpc / cu.littleh
        assert comoving.data[col].unit == u.Mpc
        assert np.all(
            np.isclose(comoving.data[col].value, scalefree.data[col].value / h)
        )


def test_parse_positions(data_path):
    dataset = read(data_path, units="scalefree")
    data = dataset.data
    cols = data.columns
    position_cols = filter(lambda col: col.split("_")[-1] in ["x", "y", "z"], cols)
    for col in position_cols:
        assert data[col].unit == u.Mpc / cu.littleh


def test_parse_mass(data_path):
    dataset = read(data_path, units="scalefree")
    data = dataset.data
    cols = data.columns
    mass_cols = filter(
        lambda col: any(part == "mass" or part[0] == "M" for part in col.split("_")),
        cols,
    )
    for col in mass_cols:
        assert data[col].unit == u.Msun / cu.littleh


def test_raw_data_is_unitless(data_path):
    """
    Make sure raw data loaded by read remains unitless even if
    the data has units.
    """
    dataset = read(data_path)
    data = dataset.data
    cols = data.columns
    assert any(data[col].unit is not None for col in cols)

    raw_data = dataset._OpenCosmoDataset__handler._InMemoryHandler__data
    assert all(raw_data[col].unit is None for col in cols)


def test_data_update_doesnt_propogate(data_path):
    dataset = read(data_path)
    data = dataset.data
    new_column = np.random.rand(len(data))
    data["new_column"] = new_column
    assert "new_column" in data.columns
    assert (
        "new_column"
        not in dataset._OpenCosmoDataset__handler._InMemoryHandler__data.columns
    )
    assert "new_column" not in dataset.data.columns


def test_unitless_convention(data_path):
    dataset = read(data_path, units="unitless")
    data = dataset.data
    cols = data.columns
    assert all(data[col].unit is None for col in cols)


def test_unit_conversion(data_path):
    dataset = read(data_path, units="unitless")
    data = dataset.data
    cols = data.columns

    unitful_dataset = dataset.with_units("comoving")
    position_cols = filter(lambda col: col.split("_")[-1] in ["x", "y", "z"], cols)

    unitless_data = dataset.data
    unitful_data = unitful_dataset.data
    for col in position_cols:
        assert unitful_data[col].unit == u.Mpc
    for col in cols:
        assert unitless_data[col].unit is None

    converted_unitless = unitful_dataset.with_units("unitless")
    converted_unitless_data = converted_unitless.data
    for col in cols:
        assert converted_unitless_data[col].unit is None


def test_invalid_unit_convention(data_path):
    with pytest.raises(ValueError):
        read(data_path, units="invalid")
