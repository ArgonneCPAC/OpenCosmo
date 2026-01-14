import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from healpy import pix2ang
from mpi4py import MPI
from pytest_mpi.parallel_assert import parallel_assert

import opencosmo as oc


@pytest.fixture
def core_path_487(diffsky_path):
    return diffsky_path / "lj_487.hdf5"


@pytest.fixture
def core_path_475(diffsky_path):
    return diffsky_path / "lj_475.hdf5"


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
    ds = ds.with_redshift_range(0.039, 0.0405)

    original_tags = ds.select("fof_halo_tag").get_data()

    original_length = len(ds)
    oc.write(path / "lightcone.hdf5", ds)
    ds = oc.open(path / "lightcone.hdf5")
    redshift_data = ds.select("redshift").data
    total_original_length = np.sum(MPI.COMM_WORLD.allgather(original_length))
    total_final_length = np.sum(MPI.COMM_WORLD.allgather(len(ds)))

    final_tags = ds.select("fof_halo_tag").get_data()

    all_original_tags = np.concatenate(MPI.COMM_WORLD.allgather(original_tags))
    all_final_tags = np.concatenate(MPI.COMM_WORLD.allgather(final_tags))

    parallel_assert(np.all(np.sort(all_original_tags) == np.sort(all_final_tags)))

    parallel_assert(redshift_data.min() >= 0.039 and redshift_data.max() <= 0.0405)
    parallel_assert(total_original_length == total_final_length)
    parallel_assert(ds.z_range == (0.039, 0.0405))


@pytest.mark.parallel(nprocs=4)
def test_diffsky_filter(core_path_487, core_path_475):
    ds = oc.open(core_path_487, core_path_475, synth_cores=True)
    original_data = ds.select("logmp0").data
    ds = ds.filter(oc.col("logmp0") > 11)
    filtered_data = ds.select("logmp0").data
    original_data = original_data[original_data.value > 11]

    assert np.all(original_data == filtered_data)


@pytest.mark.parallel(nprocs=4)
def test_diffsky_stack_with_synths(core_path_487, core_path_475, tmp_path):
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path) / "output.hdf5"
    ds = oc.open(core_path_487, core_path_475, synth_cores=True)
    ds = ds.filter(oc.col("logmp0") > 12)
    print(ds.keys())
    oc.write(tmp_path, ds)
    ds = oc.open(tmp_path, synth_cores=True)
    print(ds.keys())
    print(ds["data"])
    assert False


@pytest.mark.parallel(nprocs=4)
def test_write_some_missing(core_path_487, core_path_475, tmp_path):
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path) / "output.hdf5"
    ds = oc.open(core_path_487, core_path_475, synth_cores=False)
    if comm.Get_rank() == 0:
        ds = ds.with_redshift_range(0, 0.02)
        assert len(ds.keys()) == 1
    original_data = ds.select("early_index").get_data("numpy")
    original_data_length = comm.allgather(len(original_data))

    oc.write(tmp_path, ds)
    ds = oc.open(tmp_path, synth_cores=True)
    written_data = ds.select("early_index").get_data("numpy")

    written_data_length = comm.allgather(len(written_data))
    parallel_assert(sum(original_data_length) == sum(written_data_length))

    original_early_index = np.concatenate(comm.allgather(original_data))
    written_early_index = np.concatenate(comm.allgather(written_data))
    original_early_index.sort()
    written_early_index.sort()
    parallel_assert(np.all(original_early_index == written_early_index))


@pytest.mark.parallel(nprocs=4)
def test_lightcone_stacking(haloproperties_600_path, haloproperties_601_path, tmp_path):
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path)
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    ds = ds.take(30_000, at="random")
    for dataset in ds.values():
        assert dataset.header.lightcone["z_range"] != ds.z_range

    fof_tags = ds.select("fof_halo_tag").get_data()
    output_path = tmp_path / "data.hdf5"
    oc.write(output_path, ds)
    ds_new = oc.open(output_path)
    fof_tags_new = ds_new.select("fof_halo_tag").get_data()
    assert len(ds_new.keys()) == 1
    assert len(ds_new) == len(ds)
    assert np.all(np.unique(fof_tags) == np.unique(fof_tags_new))
    assert ds_new.z_range == ds.z_range
    assert next(iter(ds_new.values())).header.lightcone["z_range"] == ds_new.z_range
