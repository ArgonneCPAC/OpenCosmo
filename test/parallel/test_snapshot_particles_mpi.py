from __future__ import annotations

import h5py
import numpy as np
import opencosmo as oc
import pytest
from opencosmo.mpi import get_comm_world
from pytest_mpi import parallel_assert


@pytest.fixture
def snapshot_particles_path(per_test_dir, snapshot_particle_writer):
    comm = get_comm_world()
    path = per_test_dir / "snapshotparticles.hdf5"
    if comm.Get_rank() == 0:
        snapshot_particle_writer.write(path)
    comm.Barrier()
    return path


def _ids(path, name, low=None, high=None, cell_size=None) -> np.ndarray:
    with h5py.File(path, "r") as f:
        ids = f[f"{name}/data/id"][:]
        if low is None:
            return np.sort(ids)
        positions = np.stack([f[f"{name}/data/{axis}"][:] for axis in "xyz"])
    if cell_size is None:
        inside = np.all(
            (positions > low[:, None]) & (positions < high[:, None]), axis=0
        )
    else:
        cell_low = np.floor(positions / cell_size) * cell_size
        inside = np.all(
            (cell_low < high[:, None]) & (cell_low + cell_size > low[:, None]), axis=0
        )
    return np.sort(ids[inside])


@pytest.mark.parallel(nprocs=4)
def test_open_partitions_each_type(snapshot_particles_path, snapshot_particle_writer):
    comm = get_comm_world()
    collection = oc.open(snapshot_particles_path)
    for name in dict(snapshot_particle_writer.counts):
        local = collection[name].select("id").get_data("numpy")
        gathered = np.sort(np.concatenate(comm.allgather(local)))
        parallel_assert(
            np.array_equal(gathered, _ids(snapshot_particles_path, name)),
            f"{name}: ranks do not hold every particle exactly once",
        )


@pytest.mark.parallel(nprocs=4)
def test_bound(snapshot_particles_path, snapshot_particle_writer):
    comm = get_comm_world()
    low = np.array([10.0, 20.0, 5.0])
    high = np.array([30.0, 45.0, 60.0])
    collection = oc.open(snapshot_particles_path).with_units("scalefree")
    bounded = collection.bound(oc.make_box(tuple(low), tuple(high)))
    for name in dict(snapshot_particle_writer.counts):
        local = bounded[name].select("id").get_data("numpy")
        gathered = np.sort(np.concatenate(comm.allgather(local)))
        cell_size = snapshot_particle_writer.box_size / (
            2**snapshot_particle_writer.max_level
        )
        expected = _ids(snapshot_particles_path, name, low, high, cell_size)
        exact = _ids(snapshot_particles_path, name, low, high)
        parallel_assert(
            np.array_equal(gathered, expected),
            f"{name}: bounded particles differ from intersecting octree cells",
        )
        parallel_assert(
            len(exact) < len(gathered) and np.all(np.isin(exact, gathered)),
            f"{name}: snapped particles do not contain the exact selection",
        )
        parallel_assert(
            isinstance(bounded[name].index, tuple),
            f"{name}: snapped index is not chunked",
        )


@pytest.mark.parallel(nprocs=4)
def test_types_share_partition(per_test_dir, snapshot_particle_writer):
    # AGN fill only one octant, so a split of the AGN octree on its own would
    # give each rank a different part of the box than the other types.
    comm = get_comm_world()
    path = per_test_dir / "corner.hdf5"
    half = snapshot_particle_writer.box_size / 2
    if comm.Get_rank() == 0:
        snapshot_particle_writer.write(path, extent={"agn_particles": half})
    comm.Barrier()

    collection = oc.open(path).with_units("scalefree")
    octants = set()
    for name in dict(snapshot_particle_writer.counts):
        data = collection[name].select("x", "y", "z").get_data("numpy")
        cells = [(data[axis] // half).astype(int) for axis in "xyz"]
        octants |= set(zip(*cells))
    all_octants = comm.allgather(octants)
    overlap = [
        (i, j)
        for i in range(len(all_octants))
        for j in range(i + 1, len(all_octants))
        if all_octants[i] & all_octants[j]
    ]
    parallel_assert(not overlap, f"ranks {overlap} share octants")


@pytest.mark.parallel(nprocs=4)
def test_write_round_trip(snapshot_particles_path, snapshot_particle_writer):
    comm = get_comm_world()
    output = snapshot_particles_path.parent / "written.hdf5"
    collection = oc.open(snapshot_particles_path).select("id", "x", "y", "z")
    oc.write(output, collection)

    reopened = oc.open(output)
    assert isinstance(reopened, oc.ParticleCollection)
    for name in dict(snapshot_particle_writer.counts):
        assert set(reopened[name].columns) == {"id", "x", "y", "z"}
        local = reopened[name].select("id").get_data("numpy")
        gathered = np.sort(np.concatenate(comm.allgather(local)))
        parallel_assert(
            np.array_equal(gathered, _ids(snapshot_particles_path, name)),
            f"{name}: written particles differ from the input",
        )
