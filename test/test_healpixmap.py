import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.cosmology import units as cu
from numpy import random
import healsparse as hsp

import opencosmo as oc


@pytest.fixture
def healpix_map_path(map_path):
    return map_path / "test_map.hdf5"


@pytest.fixture
def all_files():
    return ["test_map.hdf5"]


@pytest.fixture
def structure_maps(map_path, all_files):
    return [map_path / f for f in all_files]


def test_open_single_map(healpix_map_path):
    c = oc.open(healpix_map_path)

def test_healpix_collection_select(healpix_map_path):
    ds = oc.open(healpix_map_path)
    columns = ds.columns
    to_select = set(['tsz','ksz'])
    ds = ds.select(to_select)
    columns_found = set(ds.columns)
    assert columns_found == to_select


def test_healpix_collection_select_healsparse(healpix_map_path):
    ds = oc.open(healpix_map_path)
    columns = ds.columns
    to_select = set(['tsz','ksz'])
    ds = ds.select(to_select)
    data = ds.get_data("healsparse")
    assert isinstance(data, dict)
    assert all(ts in data.keys() for ts in to_select)
    assert all(isinstance(col, hsp.HealSparseMap) for col in data.values())

def test_healpix_collection_select_healpix(healpix_map_path):
    ds = oc.open(healpix_map_path)
    columns = ds.columns
    to_select = set(['tsz','ksz'])
    ds = ds.select(to_select)
    data = ds.get_data("healpix")
    assert isinstance(data, dict)
    assert all(ts in data.keys() for ts in to_select)
    assert all(isinstance(col, np.ndarray) for col in data.values())


