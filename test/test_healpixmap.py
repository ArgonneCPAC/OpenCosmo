import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.cosmology import units as cu
from numpy import random

import opencosmo as oc


@pytest.fixture
def healpix_map_path(map_path):
    return map_path / "test_map_small.hdf5"

@pytest.fixture
def all_files():
    return ["test_map_small.hdf5"]

@pytest.fixture
def structure_maps(map_path, all_files):
    return [map_path / f for f in all_files]

def test_lightcone_structure_collection_open(structure_600):
    c = oc.open(*structure_maps)
    assert isinstance(c, oc.StructureCollection)

