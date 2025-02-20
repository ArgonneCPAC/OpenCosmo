import pytest
from opencosmo import cosmology
from pathlib import Path
import astropy.cosmology as ac


@pytest.fixture
def cosmology_resource_path():
    p = Path("test/resource/cosmology/")
    return p


def test_flat_lcdm(cosmology_resource_path):
    file_path = cosmology_resource_path / "flat_lcdm.hdf5"
    cosmo = cosmology.read_cosmology(file_path)
    assert isinstance(cosmo, ac.FlatLambdaCDM)
