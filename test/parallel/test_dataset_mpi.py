import numpy as np
import pytest
from opencosmo.mpi import get_comm_world
from pytest_mpi import parallel_assert

import opencosmo as oc


@pytest.fixture
def input_path(snapshot_path):
    return snapshot_path / "haloproperties.hdf5"


@pytest.mark.parallel(nprocs=4)
def test_take_global(input_path):
    comm = get_comm_world()

    ds = oc.open(input_path)
    total_length = sum(comm.allgather(len(ds)))
    n_to_take = np.random.randint(total_length // 4, int(total_length * 0.75))
    n_to_take = comm.bcast(n_to_take)

    ds = ds.take(n_to_take, mode="global")
    all_lengths = comm.allgather(len(ds))

    parallel_assert(sum(all_lengths) == n_to_take)
