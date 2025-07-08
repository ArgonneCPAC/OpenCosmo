import pytest
from astropy.table import Table

import opencosmo as oc


@pytest.fixture
def input_path(snapshot_path):
    return snapshot_path / "galaxyproperties.hdf5"


@pytest.fixture
def multi_path(snapshot_path):
    return snapshot_path / "haloproperties_multi.hdf5"


@pytest.fixture
def particle_path(snapshot_path):
    return snapshot_path / "haloparticles.hdf5"


def test_oc_open(input_path):
    dataset = oc.open(input_path)
    assert isinstance(dataset.data, Table)


def test_all_columns_included(input_path):
    dataset = oc.open(input_path)
    cols = set(dataset._Dataset__handler._DatasetHandler__group.keys())
    cols_read = set(dataset.columns)
    assert cols == cols_read


def test_read_multi(multi_path):
    dataset = oc.open(multi_path)
    assert isinstance(dataset, oc.SimulationCollection)


def test_dataset_repr(input_path):
    dataset = oc.open(input_path)
    assert isinstance(dataset.__repr__(), str)
