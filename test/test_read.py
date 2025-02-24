from pathlib import Path

import astropy.cosmology.units as cu
import astropy.units as u
import pytest
from astropy.table import Table

from opencosmo import read


@pytest.fixture
def data_path():
    return Path("test/resource/galaxyproperties.hdf5")


def test_read(data_path):
    dataset = read(data_path)
    assert isinstance(dataset.data, Table)


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
