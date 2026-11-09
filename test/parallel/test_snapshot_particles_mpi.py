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


def _ids(path, name, low=None, high=None) -> np.ndarray:
    with h5py.File(path, "r") as f:
        ids = f[f"{name}/data/id"][:]
        if low is None:
            return np.sort(ids)
        positions = np.stack([f[f"{name}/data/{axis}"][:] for axis in "xyz"])
    inside = np.all((positions > low[:, None]) & (positions < high[:, None]), axis=0)
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
        expected = _ids(snapshot_particles_path, name, low, high)
        parallel_assert(
            np.array_equal(gathered, expected),
            f"{name}: bounded particles differ from the brute-force selection",
        )
