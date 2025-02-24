from pathlib import Path

import pytest
from astropy.table import Table

from opencosmo import read


@pytest.fixture
def data_path():
    return Path("test/resource/galaxyproperties.hdf5")


def test_read(data_path):
    dataset = read(data_path)
    assert isinstance(dataset.data, Table)
