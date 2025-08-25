import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.cosmology import units as cu
from numpy import random

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
    assert ds.z_range == (0.04, 0.0405)


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


def test_lc_collection_select(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    columns = ds.columns
    to_select = set(random.choice(columns, 10))

    ds = ds.select(to_select)
    columns_found = set(ds.data.columns)

    assert columns_found == to_select


def test_lc_collection_select_numpy(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    columns = ds.columns
    to_select = set(random.choice(columns, 10))

    ds = ds.select(to_select)
    data = ds.get_data("numpy")
    assert isinstance(data, dict)
    assert all(ts in data.keys() for ts in to_select)
    assert all(isinstance(col, np.ndarray) for col in data.values())


def test_lc_collection_drop(haloproperties_600_path, haloproperties_601_path, tmp_path):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    columns = ds.columns
    to_drop = set(random.choice(columns, 10))

    ds = ds.drop(to_drop)
    columns_found = set(ds.data.columns)

    assert not columns_found.intersection(to_drop)


def test_lc_collection_take(haloproperties_600_path, haloproperties_601_path, tmp_path):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    n_to_take = int(0.75 * len(ds))
    ds_start = ds.take(n_to_take, "start")
    ds_end = ds.take(n_to_take, "end")
    ds_random = ds.take(n_to_take, "random")
    tags = ds.select("fof_halo_tag").data
    tags_start = ds_start.select("fof_halo_tag").data
    tags_end = ds_end.select("fof_halo_tag").data
    tags_random = ds_random.select("fof_halo_tag").data
    assert np.all(tags[:n_to_take] == tags_start)
    assert np.all(tags[-n_to_take:] == tags_end)
    assert len(tags_random) == n_to_take and len(set(tags_random)) == len(tags_random)


def test_lc_collection_derive(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    vsqrd = (
        oc.col("fof_halo_com_vx") ** 2
        + oc.col("fof_halo_com_vy") ** 2
        + oc.col("fof_halo_com_vz") ** 2
    )
    ke = 0.5 * oc.col("fof_halo_mass") * vsqrd
    ds = ds.with_new_columns(ke=ke)
    ke = ds.select("ke")
    assert ke.data.unit == u.solMass * u.Unit("km/s") ** 2


def test_lc_collection_add(haloproperties_600_path, haloproperties_601_path, tmp_path):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    data = np.random.randint(0, 100, len(ds)) * u.deg
    ds = ds.with_new_columns(random=data)
    stored_data = ds.select("random").get_data()
    assert np.all(stored_data == data)


def test_lc_collection_filter(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    ds = ds.filter(oc.col("fof_halo_mass") > 1e14)
    assert np.all(ds.data["fof_halo_mass"].value > 1e14)


def test_lc_collection_evaluate(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path).take(200)

    def offset(
        fof_halo_com_x,
        fof_halo_com_y,
        fof_halo_com_z,
        fof_halo_center_x,
        fof_halo_center_y,
        fof_halo_center_z,
    ):
        dx = fof_halo_com_x - fof_halo_center_x
        dy = fof_halo_com_y - fof_halo_center_y
        dz = fof_halo_com_z - fof_halo_center_z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    ds_vec = ds.evaluate(offset, vectorize=True, insert=True)
    ds_iter = ds.evaluate(offset, insert=True)

    offset_vec = ds_vec.select("offset").get_data("numpy")
    offset_iter = ds_iter.select("offset").get_data("numpy")
    assert np.all(offset_vec == offset_iter)


def test_lc_collection_evaluate_noinsert(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path).take(200)

    def offset(
        fof_halo_com_x,
        fof_halo_com_y,
        fof_halo_com_z,
        fof_halo_center_x,
        fof_halo_center_y,
        fof_halo_center_z,
    ):
        dx = fof_halo_com_x - fof_halo_center_x
        dy = fof_halo_com_y - fof_halo_center_y
        dz = fof_halo_com_z - fof_halo_center_z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    result = ds.evaluate(offset, vectorize=True, insert=False)

    assert len(result["offset"]) == len(ds)
    assert np.all(result["offset"] > 0)


def test_lc_collection_evaluate_mapped_kwarg(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path).take(200)
    random_data = np.random.randint(0, 100, len(ds))

    def offset(
        fof_halo_com_x,
        fof_halo_com_y,
        fof_halo_com_z,
        fof_halo_center_x,
        fof_halo_center_y,
        fof_halo_center_z,
        random_value,
        other_value,
    ):
        _ = fof_halo_com_x - fof_halo_center_x
        _ = fof_halo_com_y - fof_halo_center_y
        _ = fof_halo_com_z - fof_halo_center_z
        return random_value * other_value

    result = ds.evaluate(
        offset, vectorize=True, insert=False, random_value=random_data, other_value=7
    )

    assert len(result["offset"]) == len(ds)
    assert np.all(result["offset"] == random_data * 7)


def test_lc_collection_units(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds_comoving = oc.open(haloproperties_600_path, haloproperties_601_path)
    ds_scalefree = ds_comoving.with_units("scalefree")
    data_scalefree = ds_scalefree.data
    data_comoving = ds_comoving.data
    h = ds_comoving.cosmology.h

    location_columns = [f"fof_halo_com_{dim}" for dim in ("x", "y", "z")]
    for column in location_columns:
        assert data_scalefree[column].unit == u.Mpc / cu.littleh
        assert data_comoving[column].unit == u.Mpc
        assert np.all(
            np.isclose(data_scalefree[column].value, data_comoving[column].value * h)
        )
