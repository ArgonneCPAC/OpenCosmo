import os

import astropy.units as u
import healpy as hp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from healpy import pix2ang
from mpi4py import MPI
from pytest_mpi.parallel_assert import parallel_assert

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


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parallel(nprocs=4)
def test_healpix_index(healpix_map_path):
    ds = oc.open(healpix_map_path)
    pixels = ds.pixels

    pixel = np.random.choice(ds.pixels)
    center = pix2ang(ds.nside, pixel, True, True)
    radius = 2 * u.deg
    center_coord = SkyCoord(*center, unit=("deg", "deg"))

    theta, phi = hp.pix2ang(ds.nside, pixels, nest=True)
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
    parallel_assert(all(seps < radius))
    parallel_assert(len(data["tsz"].valid_pixels) == n_raw)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parallel(nprocs=4)
def test_healpix_index_chain_failure(healpix_map_path):
    ds = oc.open(healpix_map_path)

    center1 = (45 * u.deg, -45 * u.deg)
    center2 = (45 * u.deg, 45 * u.deg)
    radius = 2 * u.deg

    region1 = oc.make_cone(center1, radius)
    region2 = oc.make_cone(center2, radius)
    ds = ds.bound(region1)
    ds = ds.bound(region2)
    parallel_assert(len(ds) == 0)


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="There is a known issue with writing healpix maps after cone searches near rank boundaries.",
)
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parallel(nprocs=4)
def test_healpix_write(healpix_map_path, tmp_path):
    ds = oc.open(healpix_map_path)

    path = MPI.COMM_WORLD.bcast(tmp_path)
    # This is a known failing case
    centers = [
        (np.float64(24.741210937499996), np.float64(31.191736968831727)),
        (np.float64(31.794433593749996), np.float64(11.034862721522941)),
        (np.float64(68.31277533039648), np.float64(-63.84710895984014)),
        (np.float64(211.83837890624997), np.float64(-30.34518296958825)),
    ]

    pixel = np.random.choice(ds.pixels)
    center = pix2ang(ds.nside, pixel, True, True)
    center = centers[MPI.COMM_WORLD.Get_rank()]

    region = oc.make_cone(center, 2 * u.deg)
    ds = ds.bound(region)

    oc.write(path / "map_test.hdf5", ds)
    new_ds = oc.open(path / "map_test.hdf5")

    radius2 = 1 * u.deg
    region2 = oc.make_cone(center, radius2)
    new_ds = new_ds.bound(region2)
    ds = ds.bound(region2)

    all_valid_pixels = np.concatenate(
        MPI.COMM_WORLD.allgather(ds.data["tsz"].valid_pixels)
    )
    written_valid_pixels = np.concatenate(
        MPI.COMM_WORLD.allgather(new_ds.data["tsz"].valid_pixels)
    )

    parallel_assert(set(all_valid_pixels) == set(written_valid_pixels))


@pytest.mark.parallel(nprocs=4)
def test_healpix_partition(healpix_map_path):
    ds = oc.open(healpix_map_path)
    nside = ds.nside
    npix = hp.nside2npix(nside)
    rank = MPI.COMM_WORLD.Get_rank()
    pix_boundaries = [i * npix / 4 for i in range(5)]
    expected_pixels = np.arange(pix_boundaries[rank], pix_boundaries[rank + 1])
    assert np.all(expected_pixels == ds.pixels)


@pytest.mark.parallel(nprocs=4)
def test_write_single_map(healpix_map_path, tmp_path):
    path = MPI.COMM_WORLD.bcast(tmp_path)
    ds = oc.open(healpix_map_path)

    original_length = len(ds.data["tsz"].valid_pixels)
    oc.write(path / "map.hdf5", ds)
    ds = oc.open(path / "map.hdf5")
    parallel_assert(len(ds.data["tsz"].valid_pixels) == original_length)


@pytest.mark.parallel(nprocs=4)
def test_healpix_downgrade(healpix_map_path):
    ds = oc.open(healpix_map_path)
    downgraded_ds = ds.with_resolution(ds.nside // 16)
    data = downgraded_ds.get_data()
    expected_pixels = hp.nside2npix(ds.nside // 16)
    for map_ in data.values():
        all_pixels = np.concatenate(MPI.COMM_WORLD.allgather(map_.valid_pixels))
        assert np.all(all_pixels == np.arange(expected_pixels))


@pytest.mark.parallel(nprocs=4)
def test_healpix_downgrade_write(healpix_map_path, tmp_path):
    path = MPI.COMM_WORLD.bcast(tmp_path)
    ds = oc.open(healpix_map_path)
    original_data = ds.get_data()
    downgraded_ds = ds.with_resolution(ds.nside // 16)
    oc.write(path / "map.hdf5", downgraded_ds)
    expected_pixels = hp.nside2npix(ds.nside // 16)
    downgraded_ds = oc.open(path / "map.hdf5")
    data = downgraded_ds.get_data()
    for name, map_ in data.items():
        all_pixels = np.concatenate(MPI.COMM_WORLD.allgather(map_.valid_pixels))
        assert np.all(all_pixels == np.arange(expected_pixels))

        manually_downgraded_data = original_data[name][
            original_data[name].valid_pixels
        ].reshape((-1, (16**2))).sum(axis=1) / (16**2)
        toolkt_downgraded_data = map_[map_.valid_pixels]
        assert np.all(
            np.isclose(
                manually_downgraded_data,
                toolkt_downgraded_data,
                atol=1.0e-13,
            )
        )
