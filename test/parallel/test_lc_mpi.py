import os
import shutil

import astropy.units as u
import h5py
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from healpy import pix2ang
from mpi4py import MPI
from opencosmo.mpi import get_comm_world
from pytest_mpi.parallel_assert import parallel_assert

import opencosmo as oc

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture
def per_test_dir(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
):
    """
    Creates a unique directory for each test and deletes it after the test finishes.

    Uses tmp_path_factory so you can control base temp location via pytest's
    tempdir handling, and also so it can be used from broader-scoped fixtures
    if needed.
    """
    # request.node.nodeid is unique across parameterizations; sanitize for filesystem
    nodeid = (
        request.node.nodeid.replace("/", "_")
        .replace("::", "__")
        .replace("[", "_")
        .replace("]", "_")
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        path = tmp_path_factory.mktemp(nodeid)
    else:
        path = None
    path_to_return = comm.bcast(path)

    try:
        yield path_to_return
    finally:
        # Close out storage pressure immediately after each test
        if IN_GITHUB_ACTIONS and rank == 0:
            shutil.rmtree(path, ignore_errors=True)


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
    raw_data = ds.get_data()

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
    data = ds.bound(region).get_data()
    ra = data["phi"]
    dec = np.pi / 2 - data["theta"]

    coordinates = SkyCoord(ra, dec, unit="radian")
    coords_builtin = SkyCoord(data["ra"], data["dec"])

    seps = center_coord.separation(coordinates)
    seps = seps.to(u.degree)
    seps_builtin = center_coord.separation(coords_builtin).to(u.deg)
    parallel_assert(all(seps < radius))
    parallel_assert(all(seps_builtin < radius))
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
def test_healpix_write(haloproperties_600_path, per_test_dir):
    comm = get_comm_world()
    ds = oc.open(haloproperties_600_path)
    assert "redshift" in ds.columns

    pixel = np.random.choice(ds.region.pixels)
    center = pix2ang(ds.region.nside, pixel, True, True)

    region = oc.make_cone(center, 2 * u.deg)
    ds = ds.bound(region)

    oc.write(per_test_dir / "lightcone_test.hdf5", ds)
    new_ds = oc.open(per_test_dir / "lightcone_test.hdf5")

    radius2 = 1 * u.deg
    region2 = oc.make_cone(center, radius2)
    new_ds = new_ds.bound(region2)
    ds = ds.bound(region2)

    rank_tags = ds.select("fof_halo_tag").get_data()
    new_rank_tags = new_ds.select("fof_halo_tag").get_data()

    all_tags = np.concatenate(comm.allgather(rank_tags))
    all_new_tags = np.concatenate(comm.allgather(new_rank_tags))

    parallel_assert(np.all(np.sort(all_tags) == np.sort(all_new_tags)))


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parallel(nprocs=4)
def test_box_search(haloproperties_600_path):
    """Each rank box-searches its own pixel's neighbourhood and gets the correct rows."""
    ds = oc.open(haloproperties_600_path)
    raw_data = ds.select(("theta", "phi")).get_data("numpy")

    # Each rank picks a pixel it owns and builds a ±1° box around its centre.
    pixel = np.random.choice(ds.region.pixels)
    ra_center, dec_center = pix2ang(ds.region.nside, pixel, lonlat=True, nest=True)
    half_width = 1.0  # degrees

    ra_min = ra_center - half_width
    ra_max = ra_center + half_width
    dec_min = dec_center - half_width
    dec_max = dec_center + half_width

    p1 = SkyCoord(ra_min * u.deg, dec_min * u.deg)
    p2 = SkyCoord(ra_max * u.deg, dec_max * u.deg)

    # Expected count from manual filter of this rank's raw data.
    raw_ra = raw_data["phi"]
    raw_dec = np.pi / 2 - raw_data["theta"]
    coordinates = SkyCoord(raw_ra, raw_dec, unit="rad")

    n_expected = int(
        np.sum(
            (coordinates.ra > p1.ra)
            & (coordinates.ra < p2.ra)
            & (coordinates.dec > p1.dec)
            & (coordinates.dec < p2.dec)
        )
    )

    result = ds.box_search(p1, p2)
    parallel_assert(len(result) == n_expected)
    data = result.select(("phi", "theta")).get_data("numpy")

    result_ra = data["phi"]
    result_dec = np.pi / 2 - data["theta"]
    result_coords = SkyCoord(result_ra, result_dec, unit="rad")

    parallel_assert(np.all((result_coords.ra > p1.ra) & (result_coords.ra < p2.ra)))
    parallel_assert(np.all((result_coords.dec > p1.dec) & (result_coords.dec < p2.dec)))


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parallel(nprocs=4)
def test_box_search_zero_length(haloproperties_600_path):
    """A fixed box in a small sky region gives data on one rank and zero on the rest."""
    comm = MPI.COMM_WORLD

    ds = oc.open(haloproperties_600_path)

    # This box is known to contain data (confirmed by serial tests).
    # Because the HEALPix index is spatially partitioned across ranks, only the
    # rank that owns those pixels will return results.
    p1 = SkyCoord(43 * u.deg, -47 * u.deg)
    p2 = SkyCoord(47 * u.deg, -43 * u.deg)

    result = ds.box_search(p1, p2)
    lengths = comm.allgather(len(result))

    # Overall the query must find data.
    parallel_assert(sum(lengths) > 0)
    # And because the region is small, at least one rank owns no pixels there.
    parallel_assert(any(n == 0 for n in lengths))


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parallel(nprocs=4)
def test_box_search_chain_failure(haloproperties_600_path):
    """Chaining two disjoint box searches produces an empty result on all ranks."""
    ds = oc.open(haloproperties_600_path)

    # First box is in the data region; second is disjoint (Dec flipped to +45°).
    p1 = SkyCoord(43 * u.deg, -47 * u.deg)
    p2 = SkyCoord(47 * u.deg, -43 * u.deg)
    p3 = SkyCoord(43 * u.deg, 43 * u.deg)
    p4 = SkyCoord(47 * u.deg, 47 * u.deg)

    result = ds.box_search(p1, p2).box_search(p3, p4)
    parallel_assert(len(result) == 0)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parallel(nprocs=4)
def test_box_search_write(haloproperties_600_path, per_test_dir):
    """Written box-search result supports a narrower refinement search on re-open."""
    ds = oc.open(haloproperties_600_path)

    # Each rank works with a pixel it owns so the search is guaranteed to find data.
    pixel = np.random.choice(ds.region.pixels)
    ra_center, dec_center = pix2ang(ds.region.nside, pixel, lonlat=True, nest=True)

    # Write with a wider box, refine with a narrower one after re-open.
    p1_outer = SkyCoord((ra_center - 2.0) * u.deg, (dec_center - 2.0) * u.deg)
    p2_outer = SkyCoord((ra_center + 2.0) * u.deg, (dec_center + 2.0) * u.deg)
    ds = ds.box_search(p1_outer, p2_outer)

    oc.write(per_test_dir / "box_search_test.hdf5", ds)
    new_ds = oc.open(per_test_dir / "box_search_test.hdf5")

    p1_inner = SkyCoord((ra_center - 1.0) * u.deg, (dec_center - 1.0) * u.deg)
    p2_inner = SkyCoord((ra_center + 1.0) * u.deg, (dec_center + 1.0) * u.deg)
    ds = ds.box_search(p1_inner, p2_inner)
    new_ds = new_ds.box_search(p1_inner, p2_inner)

    assert set(ds.get_data()["fof_halo_tag"]) == set(new_ds.get_data()["fof_halo_tag"])


@pytest.mark.parallel(nprocs=4)
def test_lc_collection_write_single(
    haloproperties_600_path, haloproperties_601_path, per_test_dir
):
    comm = get_comm_world()
    ds = oc.open(haloproperties_601_path, haloproperties_600_path)
    ds = ds.with_redshift_range(0.040, 0.0405)
    original_length = comm.allreduce(len(ds))
    oc.write(per_test_dir / "lightcone.hdf5", ds)
    ds = oc.open(per_test_dir / "lightcone.hdf5")
    data = ds.select("redshift").get_data()
    new_length = comm.allreduce(len(data))

    parallel_assert(data.min() >= 0.040 and data.max() <= 0.0405)
    parallel_assert(original_length == new_length)
    parallel_assert(ds.z_range == (0.04, 0.0405))


@pytest.mark.parallel(nprocs=4)
def test_lc_collection_write(
    haloproperties_600_path, haloproperties_601_path, per_test_dir
):
    ds = oc.open(haloproperties_601_path, haloproperties_600_path)
    ds = ds.with_redshift_range(0.039, 0.0405)

    original_tags = ds.select("fof_halo_tag").get_data()

    original_length = len(ds)
    oc.write(per_test_dir / "lightcone.hdf5", ds)
    ds = oc.open(per_test_dir / "lightcone.hdf5")
    redshift_data = ds.select("redshift").get_data()
    total_original_length = np.sum(MPI.COMM_WORLD.allgather(original_length))
    total_final_length = np.sum(MPI.COMM_WORLD.allgather(len(ds)))

    final_tags = ds.select("fof_halo_tag").get_data()

    all_original_tags = np.concatenate(MPI.COMM_WORLD.allgather(original_tags))
    all_final_tags = np.concatenate(MPI.COMM_WORLD.allgather(final_tags))

    parallel_assert(np.all(np.sort(all_original_tags) == np.sort(all_final_tags)))

    parallel_assert(redshift_data.min() >= 0.039 and redshift_data.max() <= 0.0405)
    parallel_assert(total_original_length == total_final_length)
    parallel_assert(ds.z_range == (0.039, 0.0405))


class Counter:
    def __init__(self):
        self.__counts = []

    def append_count(self, count: int):
        self.__counts.append(count)

    def get_max_count(self):
        return max(self.__counts)

    @property
    def counts(self):
        return self.__counts


@pytest.mark.parallel(nprocs=4)
def test_lc_collection_batched(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    batch_size = 1000

    def offset(
        fof_halo_com_x,
        fof_halo_com_y,
        fof_halo_com_z,
        fof_halo_center_x,
        fof_halo_center_y,
        fof_halo_center_z,
        counter: Counter,
    ):
        counter.append_count(len(fof_halo_center_x))
        dx = fof_halo_com_x - fof_halo_center_x
        dy = fof_halo_com_y - fof_halo_center_y
        dz = fof_halo_com_z - fof_halo_center_z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    counter = Counter()
    offset = ds.evaluate(
        offset, vectorize=True, insert=False, batch_size=batch_size, counter=counter
    )["offset"]

    assert max(counter.counts) <= batch_size
    assert len(counter.counts) >= len(ds) // batch_size
    assert np.all(offset > 0)
    assert len(offset) == len(ds)


@pytest.mark.parallel(nprocs=4)
def test_lc_collection_batched_lazy(
    haloproperties_600_path, haloproperties_601_path, tmp_path
):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    batch_size = 1000

    def offset(
        fof_halo_com_x,
        fof_halo_com_y,
        fof_halo_com_z,
        fof_halo_center_x,
        fof_halo_center_y,
        fof_halo_center_z,
        counter: Counter,
    ):
        counter.append_count(len(fof_halo_center_x))
        dx = fof_halo_com_x - fof_halo_center_x
        dy = fof_halo_com_y - fof_halo_center_y
        dz = fof_halo_com_z - fof_halo_center_z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    counter = Counter()
    ds = ds.evaluate(
        offset, vectorize=True, insert=True, batch_size=batch_size, counter=counter
    )

    offset = ds.select("offset").get_data()
    assert max(counter.counts) <= batch_size
    assert len(counter.counts) >= len(ds) // batch_size
    assert np.all(offset > 0)
    assert len(offset) == len(ds)


@pytest.mark.parallel(nprocs=4)
def test_diffsky_filter(core_path_487, core_path_475):
    ds = oc.open(core_path_487, core_path_475, synth_cores=True)
    original_data = ds.select("logmp0").get_data()
    ds = ds.filter(oc.col("logmp0") > 11)
    filtered_data = ds.select("logmp0").get_data()
    original_data = original_data[original_data.value > 11]

    assert np.all(original_data == filtered_data)


@pytest.mark.parallel(nprocs=4)
def test_diffsky_stack_with_synths(core_path_487, core_path_475, per_test_dir):
    ds = oc.open(core_path_487, core_path_475, synth_cores=True)
    ds = ds.filter(oc.col("logmp0") > 12)
    oc.write(per_test_dir / "diffsky.hdf5", ds)
    ds = oc.open(per_test_dir / "diffsky.hdf5", synth_cores=True)


@pytest.mark.parallel(nprocs=4)
def test_write_some_missing(core_path_487, core_path_475, per_test_dir):
    comm = MPI.COMM_WORLD
    ds = oc.open(core_path_487, core_path_475, synth_cores=False)
    assert "early_index" in ds.columns
    if comm.Get_rank() == 0:
        ds = ds.with_redshift_range(0, 0.02)
        assert len(ds.keys()) == 1
    original_data = ds.select("early_index").get_data("numpy")
    original_data_length = comm.allgather(len(original_data))

    oc.write(per_test_dir / "lightcone.hdf5", ds)
    ds = oc.open(per_test_dir / "lightcone.hdf5", synth_cores=True)
    written_data = ds.select("early_index").get_data("numpy")

    written_data_length = comm.allgather(len(written_data))
    parallel_assert(sum(original_data_length) == sum(written_data_length))

    original_early_index = np.concatenate(comm.allgather(original_data))
    written_early_index = np.concatenate(comm.allgather(written_data))
    original_early_index.sort()
    written_early_index.sort()
    parallel_assert(np.all(original_early_index == written_early_index))


@pytest.mark.parallel(nprocs=4)
def test_write_diffsky_some_missing_no_stack(
    core_path_487, core_path_475, per_test_dir
):
    comm = MPI.COMM_WORLD
    ds = oc.open(core_path_475, core_path_487, synth_cores=True)
    if comm.Get_rank() == 0:
        ds.pop(475)
        assert len(ds.keys()) == 1

    columns_to_check = comm.bcast(np.random.choice(ds.columns, 10, replace=False))
    columns_to_check = np.insert(columns_to_check, 0, "gal_id")

    original_data = ds.select(columns_to_check).get_data("numpy")

    oc.write(per_test_dir / "lightcone.hdf5", ds, _min_size=10)

    ds = oc.open(per_test_dir / "lightcone.hdf5", synth_cores=True)

    written_data = ds.select(columns_to_check).get_data("numpy")

    original_galid = np.concat(comm.allgather(original_data.pop("gal_id")))
    written_galid = np.concat(comm.allgather(written_data.pop("gal_id")))
    original_order = np.argsort(original_galid)
    written_order = np.argsort(written_galid)
    columns_to_check.sort()

    for column_name in columns_to_check:
        if column_name == "gal_id":
            continue
        column_name = str(column_name)
        column_data_original = np.concat(comm.allgather(original_data.pop(column_name)))
        column_data_written = np.concat(comm.allgather(written_data.pop(column_name)))
        parallel_assert(
            np.all(
                column_data_original[original_order]
                == column_data_written[written_order]
            )
        )


@pytest.mark.parallel(nprocs=4)
def test_open_parallel_top_host(core_path_487, core_path_475):
    with h5py.File(core_path_487) as f:
        core_map = _get_expected_core_tags(f["cores"])
    with h5py.File(core_path_475) as f:
        core_map |= _get_expected_core_tags(f["cores"])

    ds = oc.open(core_path_475, core_path_487)
    data = ds.select("top_host_idx", "core_tag").get_data()

    _assert_top_host_idx_correct(data, core_map)
    _assert_all_group_members_present(data, core_map)


@pytest.mark.parallel(nprocs=4)
def test_open_write_parallel_top_host(core_path_487, core_path_475, per_test_dir):
    with h5py.File(core_path_475) as f:
        core_map = _get_expected_core_tags(f["cores"])

    with h5py.File(core_path_487) as f:
        core_map |= _get_expected_core_tags(f["cores"])

    ds = oc.open(core_path_475, core_path_487)
    data = ds.select("top_host_idx", "core_tag").get_data("numpy")

    oc.write(per_test_dir / "test.hdf5", ds)
    with h5py.File(per_test_dir / "test.hdf5") as f:
        written_core_map = _get_expected_core_tags(f["475_475"])
    assert core_map == written_core_map

    data = (
        oc.open(per_test_dir / "test.hdf5")
        .select("top_host_idx", "core_tag")
        .get_data("numpy")
    )

    _assert_top_host_idx_correct(data, core_map)
    _assert_all_group_members_present(data, core_map)


@pytest.mark.parallel(nprocs=4)
def test_open_write_parallel_top_after_filter(
    core_path_487, core_path_475, per_test_dir
):
    with h5py.File(core_path_475) as f:
        core_map = _get_expected_core_tags(f["cores"])

    with h5py.File(core_path_487) as f:
        core_map |= _get_expected_core_tags(f["cores"])

    ds = oc.open(core_path_475, core_path_487, keep_top_host=True).take(10)
    data = ds.select("top_host_idx", "core_tag").get_data("numpy")
    _assert_top_host_idx_correct(data, core_map)
    _assert_all_group_members_present(data, core_map)

    oc.write(per_test_dir / "test.hdf5", ds)

    data = (
        oc.open(per_test_dir / "test.hdf5")
        .select("top_host_idx", "core_tag")
        .get_data("numpy")
    )

    _assert_top_host_idx_correct(data, core_map)
    _assert_all_group_members_present(data, core_map)


@pytest.mark.parallel(nprocs=4)
def test_keep_top_host_filter(core_path_487, core_path_475):
    with h5py.File(core_path_487) as f:
        core_map = _get_expected_core_tags(f["cores"])
    with h5py.File(core_path_475) as f:
        core_map |= _get_expected_core_tags(f["cores"])

    ds = oc.open(core_path_475, core_path_487, keep_top_host=True)
    data = ds.take(10).select("top_host_idx", "core_tag").get_data()

    _assert_top_host_idx_correct(data, core_map)
    _assert_all_group_members_present(data, core_map)


@pytest.mark.parallel(nprocs=4)
def test_write_some_missing_no_stack(
    haloproperties_600_path, haloproperties_601_path, per_test_dir
):
    comm = MPI.COMM_WORLD
    ds = oc.open(haloproperties_601_path, haloproperties_600_path)
    if comm.Get_rank() == 0:
        ds.pop(601)
        assert len(ds.keys()) == 1

    original_halo_tags = ds.select("fof_halo_tag").get_data()

    original_data_length = comm.allgather(len(ds))
    oc.write(per_test_dir / "lightcone.hdf5", ds, _min_size=10)
    ds = oc.open(per_test_dir / "lightcone.hdf5", synth_cores=True)

    written_data_length = comm.allgather(len(ds))
    written_halo_tags = ds.select("fof_halo_tag").get_data()
    assert sum(original_data_length) == sum(written_data_length)

    original_halo_tags = np.concat(comm.allgather(original_halo_tags))
    written_halo_tags = np.concat(comm.allgather(written_halo_tags))

    assert len(np.setdiff1d(original_halo_tags, written_halo_tags)) == 0


@pytest.mark.parallel(nprocs=4)
def test_lightcone_stacking(
    haloproperties_600_path, haloproperties_601_path, per_test_dir
):
    comm = get_comm_world()
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    ds = ds.take(30_000, at="random")
    for dataset in ds.values():
        assert dataset.header.lightcone["z_range"] != ds.z_range

    fof_tags = ds.select("fof_halo_tag").get_data()
    output_path = per_test_dir / "data.hdf5"
    oc.write(output_path, ds)
    ds_new = oc.open(output_path)
    fof_tags_new = ds_new.select("fof_halo_tag").get_data()
    original_length = comm.allreduce(len(ds))
    new_length = comm.allreduce(len(ds_new))
    all_fof_tags = np.concat(comm.allgather(fof_tags))
    all_fof_tags_new = np.concat(comm.allgather(fof_tags_new))

    assert len(ds_new.keys()) == 1
    assert original_length == new_length
    assert np.all(np.isin(all_fof_tags, all_fof_tags_new))
    assert ds_new.z_range == ds.z_range
    assert next(iter(ds_new.values())).header.lightcone["z_range"] == ds_new.z_range


def _get_expected_core_tags(group):
    raw_top_host = group["data"]["top_host_idx"][:]
    core_tag = group["data"]["core_tag"][:]
    top_host_core_tag = core_tag[raw_top_host]
    return dict(zip(core_tag, top_host_core_tag))


def _assert_top_host_idx_correct(data, core_map):
    """
    Verify top_host_idx is correctly remapped in `data` (a numpy dict with
    "top_host_idx" and "core_tag" keys). Works for both local (per-rank) and
    global (gathered) data.

    Synthetic cores (core_tag == -1) are checked separately: each must point
    to its own row index. Real cores are checked against core_map.
    """
    synth_mask = data["core_tag"] == -1
    synth_indices = np.where(synth_mask)[0]
    assert np.all(data["top_host_idx"][synth_mask] == synth_indices)

    # Restrict to real cores for the map check, but dereference top_host_idx
    # against the full data so that indices into synthetic rows resolve correctly.
    real_mask = ~synth_mask
    real_top_host_idx = data["top_host_idx"][real_mask]
    real_core_tag = data["core_tag"][real_mask]

    has_top_host = real_top_host_idx >= 0
    found_top_host_core_tag = data["core_tag"][real_top_host_idx[has_top_host]]
    found_core_map = dict(zip(real_core_tag[has_top_host], found_top_host_core_tag))

    filtered_core_map = {
        key: val for key, val in core_map.items() if key in found_core_map
    }
    assert filtered_core_map == found_core_map

    should_have_core_map = {
        key: val
        for key, val in core_map.items()
        if val in data["core_tag"] and key in real_core_tag
    }
    assert should_have_core_map == found_core_map

    comm = get_comm_world()
    all_data_core_maps = comm.allgather(found_core_map)
    seen = set()
    for m in all_data_core_maps:
        assert len(seen.intersection(m.keys())) == 0
        seen |= m.keys()


def _assert_all_group_members_present(data, core_map):
    """
    Verify that for every top_host represented in the data, all rows from the
    full dataset that point to that top_host are also present.
    """
    host_to_members: dict = {}
    for ct, host_ct in core_map.items():
        host_to_members.setdefault(host_ct, set()).add(ct)

    present_core_tags = set(data["core_tag"])
    top_host_core_tags = set(data["core_tag"][data["top_host_idx"]])

    for top_host_ct in top_host_core_tags:
        expected_members = host_to_members.get(top_host_ct, set())
        missing = expected_members - present_core_tags
        assert not missing, (
            f"top_host {top_host_ct}: {len(missing)} member(s) missing from result"
        )
