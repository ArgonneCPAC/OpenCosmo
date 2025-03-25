import pytest
from astropy.table import Table

import opencosmo as oc


@pytest.fixture
def input_path(data_path):
    return data_path / "galaxyproperties.hdf5"


@pytest.fixture
def multi_path(data_path):
    return data_path / "haloproperties_multi.hdf5"


@pytest.fixture
def particle_path(data_path):
    return data_path / "haloparticles.hdf5"


def test_read(input_path):
    dataset = oc.read(input_path)
    assert isinstance(dataset.data, Table)


def test_all_columns_included(input_path):
    dataset = oc.read(input_path)
    cols = set(dataset._Dataset__handler._InMemoryHandler__data.keys())
    cols_read = set(dataset.data.columns)
    assert cols == cols_read


def test_read_multi(multi_path):
    dataset = oc.read(multi_path)
    assert isinstance(dataset, oc.collection.collection.SimulationCollection)


def test_particle_read(particle_path):
    dataset = oc.read(particle_path)
    assert isinstance(dataset, oc.collection.DataCollection)
    assert all("particle" in key for key in dataset.keys())
