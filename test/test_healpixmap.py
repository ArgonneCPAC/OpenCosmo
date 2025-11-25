import astropy.units as u
import healpy as hp
import healsparse as hsp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.cosmology import units as cu
from numpy import random

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


def test_healpix_index(healpix_map_path):
    ds = oc.open(healpix_map_path)
    raw_data = next(iter(ds.values())).get_metadata(["pixel"])

    center = (45 * u.deg, -45 * u.deg)
    radius = 2 * u.deg
    center_coord = SkyCoord(*center)

    theta, phi = hp.pix2ang(ds.nside, raw_data["pixel"], nest=True)
    raw_data_coords = SkyCoord(phi, np.pi / 2 - theta, unit="rad")

    raw_data_seps = center_coord.separation(raw_data_coords)
    n_raw = np.sum(raw_data_seps < radius)

    region = oc.make_cone(center, radius)
    data = ds.bound(region).get_data("healsparse")
    theta, phi = hp.pix2ang(ds.nside, data["tsz"].valid_pixels, nest=True)
    ra = phi
    dec = np.pi / 2 - theta
    coordinates = SkyCoord(ra, dec, unit="radian")
    seps = center_coord.separation(coordinates)
    seps = seps.to(u.degree)
    assert all(seps < radius)
    assert len(data["tsz"].valid_pixels) == n_raw


def test_healpix_downgrade(healpix_map_path):
    ds = oc.open(healpix_map_path)
    output = ds.downgrade_map(128)
    assert output.descriptions == ds.descriptions
    assert output.nside == 128
    raise ValueError("Need to implement additional checks")


def test_healpix_downgrade_doesnt_cache(healpix_map_path):
    """
    Downgrading data produces an entirely new dataset, which exists as an
    in-memory hdf5 file. We want to ensure this data is not also sent
    to the cache.
    """

    ds = oc.open(healpix_map_path)
    data = ds.get_data()
    output = ds.downgrade_map(128)

    # First dataset backed by actual file, data should be cached
    dataset = next(iter(ds.values()))
    cache = dataset._Dataset__state._DatasetState__cache
    assert len(cache.columns) > 1

    # New dataset entirely in-memory, so no cache
    downgraded_data = output.get_data()
    downgraded_dataset = next(iter(output.values()))
    cache = downgraded_dataset._Dataset__state._DatasetState__cache
    assert len(cache.columns) == 0


def test_healpix_index_chain_failure(healpix_map_path):
    ds = oc.open(healpix_map_path)

    center1 = (45 * u.deg, -45 * u.deg)
    center2 = (45 * u.deg, 45 * u.deg)
    radius = 2 * u.deg

    region1 = oc.make_cone(center1, radius)
    region2 = oc.make_cone(center2, radius)
    ds = ds.bound(region1)
    ds = ds.bound(region2)
    assert len(ds) == 0


def test_healpix_index_chain(healpix_map_path):
    ds = oc.open(healpix_map_path)
    raw_data = next(iter(ds.values())).get_metadata(["pixel"])

    center = (45 * u.deg, -45 * u.deg)
    center_coord = SkyCoord(*center)
    radius1 = 2 * u.deg
    radius2 = 1 * u.deg

    region1 = oc.make_cone(center, radius1)
    region2 = oc.make_cone(center, radius2)
    ds = ds.bound(region1)
    ds = ds.bound(region2)

    theta, phi = hp.pix2ang(ds.nside, raw_data["pixel"], nest=True)
    ra = phi
    dec = np.pi / 2 - theta
    raw_data_coords = SkyCoord(ra, dec, unit="radian")

    raw_data_seps = center_coord.separation(raw_data_coords)
    n_raw = np.sum(raw_data_seps < radius2)

    assert n_raw == len(ds)


def test_healpix_collection_bound(healpix_map_path):
    ds = oc.open(healpix_map_path)
    raw_data = next(iter(ds.values())).get_metadata(["pixel"])

    center = (45 * u.deg, -45 * u.deg)
    radius = 2 * u.deg
    center_coord = SkyCoord(*center)

    theta, phi = hp.pix2ang(ds.nside, raw_data["pixel"], nest=True)
    ra = phi
    dec = np.pi / 2 - theta
    raw_data_coords = SkyCoord(ra, dec, unit="radian")

    raw_data_seps = center_coord.separation(raw_data_coords)
    n_raw = np.sum(raw_data_seps < radius)

    region = oc.make_cone(center, radius)
    data = ds.bound(region).data
    theta, phi = hp.pix2ang(ds.nside, data["tsz"].valid_pixels, nest=True)
    ra = phi
    dec = np.pi / 2 - theta
    coordinates = SkyCoord(ra, dec, unit="radian")
    seps = center_coord.separation(coordinates)
    seps = seps.to(u.degree)
    assert all(seps < radius)
    assert len(data["tsz"].valid_pixels) == n_raw


def test_healpix_write(healpix_map_path, tmp_path):
    ds = oc.open(healpix_map_path)

    center = (45, -45)
    radius = 4 * u.deg

    region = oc.make_cone(center, radius)
    ds = ds.bound(region)

    oc.write(tmp_path / "map_test.hdf5", ds)
    new_ds = oc.open(tmp_path / "map_test.hdf5")

    radius2 = 2 * u.deg
    region2 = oc.make_cone(center, radius2)
    new_ds = new_ds.bound(region2)
    ds = ds.bound(region2)

    assert set(ds.data["tsz"].valid_pixels) == set(new_ds.data["tsz"].valid_pixels)


def test_healpix_write_fail(healpix_map_path, tmp_path):
    ds = oc.open(healpix_map_path)

    center = (45 * u.deg, -45 * u.deg)
    radius = 2 * u.deg

    region = oc.make_cone(center, radius)
    ds = ds.bound(region)

    oc.write(tmp_path / "map_test.hdf5", ds)
    new_ds = oc.open(tmp_path / "map_test.hdf5")

    center2 = (45 * u.deg, 45 * u.deg)
    region2 = oc.make_cone(center2, radius)
    new_ds = new_ds.bound(region2)
    assert len(new_ds) == 0


def test_healpix_collection_drop(healpix_map_path):
    ds = oc.open(healpix_map_path)
    columns = ds.columns
    to_drop = set(["tsz"])

    ds = ds.drop(to_drop)
    columns_found = set(ds.data.keys())

    assert not columns_found.intersection(to_drop)


def test_healpix_collection_take(healpix_map_path):
    ds = oc.open(healpix_map_path)
    n_to_take = int(0.75 * len(ds))
    ds_start = ds.take(n_to_take, "start")
    ds_end = ds.take(n_to_take, "end")
    ds_random = ds.take(n_to_take, "random")
    tags = ds.select("tsz").data["tsz"].valid_pixels
    tags_start = ds_start.select("tsz").data["tsz"].valid_pixels
    tags_end = ds_end.select("tsz").data["tsz"].valid_pixels
    tags_random = ds_random.select("tsz").data["tsz"].valid_pixels
    assert np.all(tags[:n_to_take] == tags_start)
    assert np.all(tags[-n_to_take:] == tags_end)
    assert len(tags_random) == n_to_take and len(set(tags_random)) == len(tags_random)


def test_healpix_collection_range(healpix_map_path):
    ds = oc.open(healpix_map_path)
    start = int(0.25 * len(ds))
    end = int(0.75 * len(ds))

    ds_range = ds.take_range(start, end)
    halo_tags = ds.select("tsz").get_data("healsparse")["tsz"].valid_pixels[start:end]
    range_halo_tags = ds_range.select("tsz").get_data("healsparse")["tsz"].valid_pixels
    assert np.all(halo_tags == range_halo_tags)


def test_open_single_map(healpix_map_path):
    c = oc.open(healpix_map_path)


def test_healpix_collection_select(healpix_map_path):
    ds = oc.open(healpix_map_path)
    columns = ds.columns
    to_select = set(["tsz", "ksz"])
    ds = ds.select(to_select)
    columns_found = set(ds.columns)
    assert columns_found == to_select


def test_healpix_collection_select_healsparse(healpix_map_path):
    ds = oc.open(healpix_map_path)
    columns = ds.columns
    to_select = set(["tsz", "ksz"])
    ds = ds.select(to_select)
    data = ds.get_data("healsparse")
    assert isinstance(data, dict)
    assert all(ts in data.keys() for ts in to_select)
    assert all(isinstance(col, hsp.HealSparseMap) for col in data.values())


def test_healpix_collection_select_healpix(healpix_map_path):
    ds = oc.open(healpix_map_path)
    columns = ds.columns
    to_select = set(["tsz", "ksz"])
    ds = ds.select(to_select)
    data = ds.get_data("healpix")
    assert isinstance(data, dict)
    assert all(ts in data.keys() for ts in to_select)
    assert all(isinstance(col, np.ndarray) for col in data.values())


def test_healpix_collection_derive(healpix_map_path):
    ds = oc.open(healpix_map_path)
    sz_sqrd = oc.col("tsz") ** 2 + oc.col("ksz") ** 2
    ds = ds.with_new_columns(weird_sz=sz_sqrd)
    weird = ds.select("weird_sz").data
    assert isinstance(weird, dict)


def test_healpix_collection_add(healpix_map_path):
    ds = oc.open(healpix_map_path)
    map_data = ds.get_data("healsparse")["tsz"].valid_pixels
    data = np.zeros(len(map_data))
    ds = ds.with_new_columns(random=data)
    stored_data = ds.select("random").get_data("healpix")["random"]
    assert np.all(stored_data == data)


def test_healpix_collection_add_sparse(healpix_map_path):
    ds = oc.open(healpix_map_path)
    map_data = ds.get_data("healsparse")["tsz"].valid_pixels
    data = np.zeros(len(map_data))
    ds = ds.with_new_columns(random=data)
    stored_data = (
        ds.select("random").get_data("healsparse")["random"].get_values_pix(map_data)
    )
    assert np.all(stored_data == data)


def test_healpix_collection_filter(healpix_map_path):
    ds = oc.open(healpix_map_path)
    ds = ds.filter(oc.col("tsz") > 0)
    hsp_map = ds.get_data("healsparse")["tsz"]
    valid_pix = hsp_map.valid_pixels
    assert np.all(hsp_map.get_values_pix(valid_pix) > 0)


def test_healpix_collection_evaluate(healpix_map_path):
    ds = oc.open(healpix_map_path).take(200)

    def offset(tsz, ksz):
        dx = tsz - ksz
        return np.sqrt(dx**2)

    ds_vec = ds.evaluate(offset, vectorize=True, insert=True)
    ds_iter = ds.evaluate(offset, insert=True)

    offset_vec = ds_vec.select("offset").get_data("healsparse")["offset"]
    offset_vec = offset_vec.get_values_pix(offset_vec.valid_pixels)
    offset_iter = ds_iter.select("offset").get_data("healsparse")["offset"]
    offset_iter = offset_iter.get_values_pix(offset_iter.valid_pixels)
    assert np.all(offset_vec == offset_iter)


def test_healpix_collection_evaluate_noinsert(healpix_map_path):
    ds = oc.open(healpix_map_path).take(200)

    def offset(tsz, ksz):
        dx = tsz - ksz
        return np.sqrt(dx**2)

    result = ds.evaluate(offset, vectorize=True, insert=False)

    assert len(result["offset"]) == 200
    assert np.all(result["offset"] >= 0)


def test_write_single_map(healpix_map_path, tmp_path):
    ds = oc.open(healpix_map_path)
    assert isinstance(ds, oc.HealpixMap)
    oc.write(tmp_path / "temp.hdf5", ds)
    ds_written = oc.open(tmp_path / "temp.hdf5")
    assert isinstance(ds, oc.HealpixMap)
    assert len(ds) == len(ds_written)
