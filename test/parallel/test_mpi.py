import opencosmo as oc
import pytest
from pytest_mpi.parallel_assert import parallel_assert
import mpi4py
import numpy as np


@pytest.fixture
def input_path(data_path):
    return data_path / "haloproperties.hdf5"


@pytest.mark.parallel(nprocs=4)
def test_mpi(input_path):
    with oc.open(input_path) as f:
        data = f.data

    parallel_assert(lambda: len(data) != 0)


@pytest.mark.parallel(nprocs=4)
def test_take(input_path):
    ds = oc.open(input_path)

    comm = mpi4py.MPI.COMM_WORLD
    data = ds.data

    rank_length = len(data)
    total_length = comm.allreduce(rank_length, op=mpi4py.MPI.SUM)
    # get a random number between 0 and total_length
    if comm.Get_rank() == 0:
        n = np.random.randint(0, total_length)
    else:
        n = 0
    n = comm.bcast(n, root=0)

    ds = ds.take(n, "random")
    data = ds.data
    ds.close()
    lengths = comm.gather(len(data), root=0)
    if comm.Get_rank() == 0:
        total_length = sum(lengths)
        assert sum(lengths) == total_length
    parallel_assert(lambda: total_length == n, participating=comm.Get_rank() == 0)


@pytest.mark.parallel(nprocs=4)
def test_filters(input_path):
    ds = oc.open(input_path)
    ds = ds.filter(oc.col("sod_halo_mass") > 0)
    data = ds.data
    ds.close()
    parallel_assert(lambda: len(data) != 0)
    parallel_assert(lambda: all(data["sod_halo_mass"] > 0))


@pytest.mark.parallel(nprocs=4)
def test_filter_write(input_path, tmp_path):
    ds = oc.open(input_path)
    ds = ds.filter(oc.col("sod_halo_mass") > 0)
    output_path = tmp_path / "filtered.hdf5"
    oc.write(output_path, ds)
    ds.close()
    ds = oc.read(output_path)
    data = ds.data
    parallel_assert(lambda: len(data) != 0)
    parallel_assert(lambda: all(data["sod_halo_mass"] > 0))
