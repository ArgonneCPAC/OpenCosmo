from pathlib import Path

import pytest
from astropy.cosmology import FlatLambdaCDM

from opencosmo.header import read_header


@pytest.fixture
def header_resource_path():
    p = Path("test/resource/flat_lcdm.hdf5")
    return p


def test_read_header(header_resource_path):
    header = read_header(header_resource_path)
    assert isinstance(header.cosmology, FlatLambdaCDM)
