import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

import opencosmo as oc


@pytest.fixture
def haloproperties_600_path(lightcone_path):
    return lightcone_path / "step_600" / "haloproperties.hdf5"


@pytest.fixture
def haloproperties_601_path(lightcone_path):
    return lightcone_path / "step_601" / "haloproperties.hdf5"


def test_healpix_index(haloproperties_600_path):
    ds = oc.open(haloproperties_600_path)
    raw_data = ds.data

    center = (45 * u.deg, -45 * u.deg)
    radius = 2 * u.deg
    center_coord = SkyCoord(*center)

    raw_data_coords = SkyCoord(
        raw_data["phi"], np.pi / 2 - raw_data["theta"], unit="rad"
    )
    raw_data_seps = center_coord.separation(raw_data_coords)
    n_raw = np.sum(raw_data_seps < radius)

    region = oc.make_cone(center, radius)
    data = ds.bound(region).data
    ra = data["phi"]
    dec = np.pi / 2 - data["theta"]

    coordinates = SkyCoord(ra, dec, unit="radian")
    seps = center_coord.separation(coordinates)
    seps = seps.to(u.degree)
    assert all(seps < radius)
    assert len(data) == n_raw


def test_healpix_index_chain_failure(haloproperties_600_path):
    ds = oc.open(haloproperties_600_path)

    center1 = (45 * u.deg, -45 * u.deg)
    center2 = (45 * u.deg, 45 * u.deg)
    radius = 2 * u.deg

    region1 = oc.make_cone(center1, radius)
    region2 = oc.make_cone(center2, radius)
    ds = ds.bound(region1)
    ds = ds.bound(region2)
    assert len(ds) == 0


def test_healpix_index_chain(haloproperties_600_path):
    ds = oc.open(haloproperties_600_path)
    raw_data = ds.data

    center = (45 * u.deg, -45 * u.deg)
    center_coord = SkyCoord(*center)
    radius1 = 2 * u.deg
    radius2 = 1 * u.deg

    region1 = oc.make_cone(center, radius1)
    region2 = oc.make_cone(center, radius2)
    ds = ds.bound(region1)
    ds = ds.bound(region2)

    raw_data_coords = SkyCoord(
        raw_data["phi"], np.pi / 2 - raw_data["theta"], unit="rad"
    )
    raw_data_seps = center_coord.separation(raw_data_coords)
    n_raw = np.sum(raw_data_seps < radius2)

    assert n_raw == len(ds)


def test_healpix_write(haloproperties_600_path, tmp_path):
    ds = oc.open(haloproperties_600_path)

    center = (45, -45)
    radius = 4 * u.deg

    region = oc.make_cone(center, radius)
    ds = ds.bound(region)

    oc.write(tmp_path / "lightcone_test.hdf5", ds)
    new_ds = oc.open(tmp_path / "lightcone_test.hdf5")

    radius2 = 2 * u.deg
    region2 = oc.make_cone(center, radius2)
    new_ds = new_ds.bound(region2)
    ds = ds.bound(region2)

    assert set(ds.data["fof_halo_tag"]) == set(new_ds.data["fof_halo_tag"])


def test_healpix_write_fail(haloproperties_600_path, tmp_path):
    ds = oc.open(haloproperties_600_path)

    center = (45 * u.deg, -45 * u.deg)
    radius = 2 * u.deg

    region = oc.make_cone(center, radius)
    ds = ds.bound(region)

    oc.write(tmp_path / "lightcone_test.hdf5", ds)
    new_ds = oc.open(tmp_path / "lightcone_test.hdf5")

    center2 = (45 * u.deg, 45 * u.deg)
    region2 = oc.make_cone(center2, radius)
    new_ds = new_ds.bound(region2)
    assert len(new_ds) == 0


def test_lightcone_physical_units(haloproperties_600_path):
    ds_comoving = oc.open(haloproperties_600_path)
    ds_physical = ds_comoving.with_units("physical")
    data_comoving = ds_comoving.data
    data_physical = ds_physical.data
    assert np.all(
        data_physical["fof_halo_com_x"]
        == (data_comoving["fof_halo_com_x"] * data_comoving["fof_halo_center_a"])
    )
    assert np.all(
        data_physical["fof_halo_com_vx"]
        == (data_comoving["fof_halo_com_vx"] * data_comoving["fof_halo_center_a"])
    )


def test_lc_collection_restrict_z(haloproperties_600_path, haloproperties_601_path):
    ds = oc.open(haloproperties_601_path, haloproperties_600_path)
    original_redshifts = ds.select("redshift").data
    ds = ds.with_redshift_range(0.040, 0.0405)
    redshifts = ds.select("redshift").data
    masked_redshifts = (original_redshifts > 0.04) & (original_redshifts < 0.0405)
    assert np.all((redshifts > 0.04) & (redshifts < 0.0405))
    assert np.sum(masked_redshifts) == len(redshifts)


def test_lc_collection_write(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_601_path, haloproperties_600_path)
    ds = ds.with_redshift_range(0.040, 0.0405)
    original_length = len(ds)
    oc.write(tmp_path / "lightcone.hdf5", ds)
    ds = oc.open(tmp_path / "lightcone.hdf5")
    data = ds.select("redshift").data
    assert data.min() >= 0.04 and data.max() <= 0.0405
    assert len(data) == original_length


def test_lc_collection_bound(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    raw_data = ds.data

    center = (45 * u.deg, -45 * u.deg)
    radius = 2 * u.deg
    center_coord = SkyCoord(*center)

    raw_data_coords = SkyCoord(
        raw_data["phi"], np.pi / 2 - raw_data["theta"], unit="rad"
    )
    raw_data_seps = center_coord.separation(raw_data_coords)
    n_raw = np.sum(raw_data_seps < radius)

    region = oc.make_cone(center, radius)
    data = ds.bound(region).data
    ra = data["phi"]
    dec = np.pi / 2 - data["theta"]

    coordinates = SkyCoord(ra, dec, unit="radian")
    seps = center_coord.separation(coordinates)
    seps = seps.to(u.degree)
    assert all(seps < radius)
    assert len(data) == n_raw
