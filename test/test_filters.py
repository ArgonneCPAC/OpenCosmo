import pytest

from opencosmo import col, read


@pytest.fixture
def input_path(data_path):
    return data_path / "haloproperties.hdf5"


def test_filters(input_path):
    ds = read(input_path)
    ds = ds.filter(col("sod_halo_mass") > 0)
    data = ds.data
    assert data["sod_halo_mass"].min() > 0


def test_multi_filters_single_column(input_path):
    ds = read(input_path)
    sod_mass = ds.data["sod_halo_mass"]
    sod_mass_unit = sod_mass.unit
    max_mass = 0.95 * sod_mass.max() * sod_mass_unit

    ds = ds.filter(col("sod_halo_mass") > 0, col("sod_halo_mass") < max_mass)
    data = ds.data
    assert data["sod_halo_mass"].min() > 0
    assert data["sod_halo_mass"].max() < max_mass.value
