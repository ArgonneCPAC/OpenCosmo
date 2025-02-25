from pathlib import Path

import astropy.cosmology.units as cu
import astropy.units as u
import numpy as np
import pytest

from opencosmo import read


@pytest.fixture
def data_path():
    return Path("test/resource/galaxyproperties.hdf5")


def test_parse_velocities(data_path):
    dataset = read(data_path)
    data = dataset.data
    cols = data.columns
    velocity_cols = filter(lambda col: col.upper()[-2:] in ["VX", "VY", "VZ"], cols)
    for col in velocity_cols:
        assert data[col].unit == u.km / u.s


def test_parse_mass(data_path):
    dataset = read(data_path)
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
