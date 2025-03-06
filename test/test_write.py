from pathlib import Path

import astropy.cosmology as ac
import h5py
import pytest

from opencosmo import read_header
from opencosmo.header import write_header


@pytest.fixture
def cosmology_resource_path(data_path):
    p = data_path / "header.hdf5"
    return p

def test_write_header(data_path, tmp_path):
    header = read_header(data_path / "header.hdf5")
    new_path = tmp_path / "header.hdf5"
    write_header(new_path, header)
    
    new_header = read_header(new_path)
    assert header.simulation == new_header.simulation
    assert header._OpenCosmoHeader__reformat_pars == new_header._OpenCosmoHeader__reformat_pars
    assert header._OpenCosmoHeader__cosmotools_pars == new_header._OpenCosmoHeader__cosmotools_pars



