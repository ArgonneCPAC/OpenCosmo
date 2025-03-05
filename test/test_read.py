import pytest
from astropy.table import Table

from opencosmo import read


@pytest.fixture
def input_path(data_path):
    return data_path / "galaxyproperties.hdf5"


def test_read(input_path):
    dataset = read(input_path)
    assert isinstance(dataset.data, Table)


def test_all_columns_included(input_path):
    dataset = read(input_path)
    cols = set(dataset._Dataset__handler._InMemoryHandler__data.keys())
    cols_read = set(dataset.data.columns)
    assert cols == cols_read
