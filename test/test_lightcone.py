import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

import opencosmo as oc


@pytest.fixture
def haloproperties_path(lightcone_path):
    return lightcone_path / "step_600" / "haloproperties.hdf5"


def test_healpix_index(haloproperties_path):
    ds = oc.open(haloproperties_path)
    raw_data = ds.data

    center = (45 * u.deg, -45 * u.deg)
    radius = 2 * u.deg
    center_coord = SkyCoord(*center)

    raw_data_coords = SkyCoord(
        raw_data["phi"], np.pi / 2 - raw_data["theta"], unit="rad"
    )
    raw_data_seps = center_coord.separation(raw_data_coords)
    n_raw = np.sum(raw_data_seps < radius)

    region = oc.Cone(center, radius)
    data = ds.bound(region).data
    ra = data["phi"]
    dec = np.pi / 2 - data["theta"]

    coordinates = SkyCoord(ra, dec, unit="radian")
    seps = center_coord.separation(coordinates)
    seps = seps.to(u.degree)
    assert all(seps < radius)
    assert len(data) == n_raw


def test_healpix_index_chain_failure(haloproperties_path):
    ds = oc.open(haloproperties_path)

    center1 = (45 * u.deg, -45 * u.deg)
    center2 = (45 * u.deg, 45 * u.deg)
    radius = 2 * u.deg

    region1 = oc.Cone(center1, radius)
    region2 = oc.Cone(center2, radius)
    ds = ds.bound(region1)
    with pytest.raises(ValueError):
        ds = ds.bound(region2)


def test_healpix_index_chain(haloproperties_path):
    ds = oc.open(haloproperties_path)
    raw_data = ds.data

    center = (45 * u.deg, -45 * u.deg)
    center_coord = SkyCoord(*center)
    radius1 = 2 * u.deg
    radius2 = 1 * u.deg

    region1 = oc.Cone(center, radius1)
    region2 = oc.Cone(center, radius2)
    ds = ds.bound(region1)
    ds = ds.bound(region2)

    raw_data_coords = SkyCoord(
        raw_data["phi"], np.pi / 2 - raw_data["theta"], unit="rad"
    )
    raw_data_seps = center_coord.separation(raw_data_coords)
    n_raw = np.sum(raw_data_seps < radius2)

    assert n_raw == len(ds)


def test_healpix_write(haloproperties_path, tmp_path):
    ds = oc.open(haloproperties_path)

    center = (45, -45)
    radius = 4 * u.deg

    region = oc.Cone(center, radius)
    ds = ds.bound(region)

    oc.write(tmp_path / "lightcone_test.hdf5", ds)
    new_ds = oc.open(tmp_path / "lightcone_test.hdf5")

    radius2 = 2 * u.deg
    region2 = oc.Cone(center, radius2)
    new_ds = new_ds.bound(region2)
    ds = ds.bound(region2)

    assert set(ds.data["fof_halo_tag"]) == set(new_ds.data["fof_halo_tag"])


def test_healpix_write_fail(haloproperties_path, tmp_path):
    ds = oc.open(haloproperties_path)

    center = (45 * u.deg, -45 * u.deg)
    radius = 2 * u.deg

    region = oc.Cone(center, radius)
    ds = ds.bound(region)

    oc.write(tmp_path / "lightcone_test.hdf5", ds)
    new_ds = oc.open(tmp_path / "lightcone_test.hdf5")

    center2 = (45 * u.deg, 45 * u.deg)
    region2 = oc.Cone(center2, radius)
    with pytest.raises(ValueError):
        new_ds = new_ds.bound(region2)
