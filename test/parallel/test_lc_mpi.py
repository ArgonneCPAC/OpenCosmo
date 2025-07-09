import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from healpy import pix2ang
from mpi4py import MPI
from pytest_mpi.parallel_assert import parallel_assert

import opencosmo as oc


@pytest.fixture
def haloproperties_600_path(lightcone_path):
    return lightcone_path / "step_600" / "haloproperties.hdf5"


@pytest.fixture
def haloproperties_601_path(lightcone_path):
    return lightcone_path / "step_601" / "haloproperties.hdf5"


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parallel(nprocs=4)
def test_healpix_index(haloproperties_600_path):
    ds = oc.open(haloproperties_600_path)
    raw_data = ds.data

    pixel = np.random.choice(ds.region.pixels)
    center = pix2ang(ds.region.nside, pixel, True, True)

    radius = 2 * u.deg
    center_coord = SkyCoord(*center, unit="deg")

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
    parallel_assert(all(seps < radius))
    parallel_assert(len(data) == n_raw)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parallel(nprocs=4)
def test_healpix_index_chain_failure(haloproperties_600_path):
    ds = oc.open(haloproperties_600_path)

    center1 = (45 * u.deg, -45 * u.deg)
    center2 = (45 * u.deg, 45 * u.deg)
    radius = 2 * u.deg

    region1 = oc.make_cone(center1, radius)
    region2 = oc.make_cone(center2, radius)
    ds = ds.bound(region1).bound(region2)
    parallel_assert(len(ds) == 0)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parallel(nprocs=4)
def test_healpix_write(haloproperties_600_path, tmp_path):
    ds = oc.open(haloproperties_600_path)
    path = MPI.COMM_WORLD.bcast(tmp_path)

    pixel = np.random.choice(ds.region.pixels)
    center = pix2ang(ds.region.nside, pixel, True, True)

    region = oc.make_cone(center, 2 * u.deg)
    ds = ds.bound(region)

    oc.write(path / "lightcone_test.hdf5", ds)
    new_ds = oc.open(path / "lightcone_test.hdf5")

    radius2 = 2 * u.deg
    region2 = oc.make_cone(center, radius2)
    new_ds = new_ds.bound(region2)
    ds = ds.bound(region2)

    assert set(ds.data["fof_halo_tag"]) == set(new_ds.data["fof_halo_tag"])


@pytest.mark.parallel(nprocs=4)
def test_lc_collection_write_single(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    path = MPI.COMM_WORLD.bcast(tmp_path)
    ds = oc.open(haloproperties_601_path, haloproperties_600_path)
    ds = ds.with_redshift_range(0.040, 0.0405)
    original_length = len(ds)
    oc.write(path / "lightcone.hdf5", ds)
    ds = oc.open(path / "lightcone.hdf5")
    data = ds.select("redshift").data
    parallel_assert(data.min() >= 0.040 and data.max() <= 0.0405)
    parallel_assert(len(data) == original_length)
    parallel_assert(ds.z_range == (0.04, 0.0405))


@pytest.mark.parallel(nprocs=4)
def test_lc_collection_write(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    path = MPI.COMM_WORLD.bcast(tmp_path)
    ds = oc.open(haloproperties_601_path, haloproperties_600_path)
    print(len(ds))
    ds = ds.with_redshift_range(0.039, 0.0405)
    original_length = len(ds)
    oc.write(path / "lightcone.hdf5", ds)
    ds = oc.open(path / "lightcone.hdf5")
    print(original_length)
    print(len(ds))
    data = ds.select("redshift").data
    parallel_assert(data.min() >= 0.039 and data.max() <= 0.0405)
    parallel_assert(len(data) == original_length)
    parallel_assert(ds.z_range == (0.039, 0.0405))
