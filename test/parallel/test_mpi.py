from pathlib import Path

import h5py
import mpi4py
import numpy as np
import pytest
from pytest_mpi.parallel_assert import parallel_assert

import opencosmo as oc


@pytest.fixture
def input_path(data_path):
    return data_path / "haloproperties.hdf5"


@pytest.fixture
def particle_path(data_path):
    return data_path / "haloparticles.hdf5"


@pytest.fixture
def malformed_header_path(input_path, tmp_path):
    update = {"n_dm": "foo"}
    return update_simulation_parameter(input_path, update, tmp_path, "malformed_header")


def update_simulation_parameter(
    base_cosmology_path: Path, parameters: dict[str, float], tmp_path: Path, name: str
):
    # make a copy of the original data
    path = tmp_path / f"{name}.hdf5"
    with h5py.File(base_cosmology_path, "r") as f:
        with h5py.File(path, "w") as file:
            f.copy(f["header"], file, "header")
            # update the attributes
            for key, value in parameters.items():
                file["header"]["simulation"]["parameters"].attrs[key] = value
    return path


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
    comm = mpi4py.MPI.COMM_WORLD
    temporary_path = tmp_path / "filtered.hdf5"
    temporary_path = comm.bcast(temporary_path, root=0)

    ds = oc.open(input_path)
    ds = ds.filter(oc.col("sod_halo_mass") > 0)
    oc.write(temporary_path, ds)
    data = ds.collect().data
    ds.close()

    ds = oc.read(temporary_path)
    written_data = ds.data
    parallel_assert(lambda: np.all(data == written_data))


@pytest.mark.parallel(nprocs=4)
def test_collect(input_path):
    with oc.open(input_path) as f:
        ds = f.filter(oc.col("sod_halo_mass") > 0).take(100, at="random").collect()

    parallel_assert(lambda: len(ds.data) == 100)


@pytest.mark.parallel(nprocs=4)
def test_select_collect(input_path):
    with oc.open(input_path) as f:
        ds = (
            f.filter(oc.col("sod_halo_mass") > 0)
            .select(["sod_halo_mass", "fof_halo_mass"])
            .take(100, at="random")
            .collect()
        )

    parallel_assert(lambda: len(ds.data) == 100)
    parallel_assert(lambda: set(ds.data.columns) == {"sod_halo_mass", "fof_halo_mass"})


@pytest.mark.parallel(nprocs=4)
def test_take_empty_rank(input_path):
    comm = mpi4py.MPI.COMM_WORLD
    with oc.open(input_path) as f:
        ds = f.filter(oc.col("sod_halo_mass") > 0)
        data = ds.data
        length = comm.allgather(len(data))
        n_to_take = length[0] + length[1] // 2
        if comm.Get_rank() in [0, 1]:
            ds = ds.take(n_to_take, at="start")
        else:
            with pytest.raises(ValueError):
                ds = ds.take(n_to_take, at="start")


@pytest.mark.parallel(nprocs=4)
def test_read_particles(particle_path):
    with oc.open(particle_path) as f:
        parallel_assert(lambda: isinstance(f, dict))


@pytest.mark.parallel(nprocs=4)
def test_write_particles(particle_path, tmp_path):
    comm = mpi4py.MPI.COMM_WORLD
    output_path = tmp_path / "particles.hdf5"
    output_path = comm.bcast(output_path, root=0)
    with oc.open(particle_path) as f:
        oc.write(output_path, f)
        data = f.collect()
        header = f._header
    if comm.Get_rank() == 0:
        read_data = oc.read(output_path)
        for key in data.keys():
            assert np.all(data[key].data == read_data[key].data)
        read_header = read_data._header
        models = ["file_pars", "simulation_pars", "reformat_pars", "cosmotools_pars"]
        for model in models:
            key = f"_OpenCosmoHeader__{model}"
            assert getattr(header, key) == getattr(read_header, key)

