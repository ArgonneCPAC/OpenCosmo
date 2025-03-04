import pytest
from astropy.table import Table

from opencosmo import read


@pytest.fixture
def input_path(data_path):
    return data_path / "galaxyproperties.hdf5"


def test_read(input_path):
    dataset = read(input_path)
    assert isinstance(dataset.data, Table)
