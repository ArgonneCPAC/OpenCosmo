from collections import defaultdict
from pathlib import Path

import h5py
import mpi4py
import numpy as np
import pytest
from pytest_mpi.parallel_assert import parallel_assert

import opencosmo as oc
from opencosmo.link import open_linked_files


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


@pytest.fixture
def all_paths(data_path: Path):
    files = ["haloparticles.hdf5", "haloproperties.hdf5", "sodproperties.hdf5"]

    hdf_files = [data_path / file for file in files]
    return list(hdf_files)


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

    parallel_assert(len(data) != 0)


@pytest.mark.parallel(nprocs=4)
def test_take(input_path):
    ds = oc.open(input_path)
    data = ds.data
    n = 1000
    ds = ds.take(n, "random")
    data = ds.data
    ds.close()
    parallel_assert(len(data) == n)

    halo_tags = data["fof_halo_tag"]
    gathered_tags = mpi4py.MPI.COMM_WORLD.gather(halo_tags, root=0)
    tags = set()
    if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
        for tag_list in gathered_tags:
            tags.update(tag_list)

        assert len(tags) == 4 * n


@pytest.mark.parallel(nprocs=4)
def test_filters(input_path):
    ds = oc.open(input_path)
    ds = ds.filter(oc.col("sod_halo_mass") > 0)
    data = ds.data
    ds.close()
    parallel_assert(len(data) != 0)
    parallel_assert(all(data["sod_halo_mass"] > 0))


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

    handler = ds._Dataset__handler
    tree = handler._InMemoryHandler__tree
    starts = tree._Tree__starts
    sizes = tree._Tree__sizes
    parallel_assert(np.all(data == written_data))
    for level in sizes:
        parallel_assert(np.sum(sizes[level]) == len(handler))
        parallel_assert(starts[level][0] == 0)
        if level > 0:
            sizes_from_starts = np.diff(np.append(starts[level], len(handler)))
            parallel_assert(np.all(sizes_from_starts == sizes[level]))


@pytest.mark.parallel(nprocs=4)
def test_collect(input_path):
    with oc.open(input_path) as f:
        ds = f.filter(oc.col("sod_halo_mass") > 0).take(100, at="random").collect()

    parallel_assert(len(ds.data) == 400)


@pytest.mark.parallel(nprocs=4)
def test_select_collect(input_path):
    with oc.open(input_path) as f:
        ds = (
            f.filter(oc.col("sod_halo_mass") > 0)
            .select(["sod_halo_mass", "fof_halo_mass"])
            .take(100, at="random")
            .collect()
        )

    parallel_assert(len(ds.data) == 400)
    parallel_assert(set(ds.data.columns) == {"sod_halo_mass", "fof_halo_mass"})


@pytest.mark.parallel(nprocs=4)
def test_read_particles(particle_path):
    with oc.open(particle_path) as f:
        parallel_assert(isinstance(f, dict))


@pytest.mark.parallel(nprocs=4)
def test_write_particles(particle_path, tmp_path):
    comm = mpi4py.MPI.COMM_WORLD
    output_path = tmp_path / "particles.hdf5"
    output_path = comm.bcast(output_path, root=0)
    with oc.open(particle_path) as f:
        oc.write(output_path, f)
    original_data = oc.read(particle_path)
    written_data = oc.read(output_path)
    indices = np.random.randint(0, len(original_data), 100)
    for key in original_data.keys():
        assert np.all(
            original_data[key].data[indices] == written_data[key].data[indices]
        )
    header = original_data.header
    written_header = written_data.header
    models = ["file_pars", "simulation_pars", "reformat_pars", "cosmotools_pars"]
    for model in models:
        key = f"_OpenCosmoHeader__{model}"
        parallel_assert(getattr(header, key) == getattr(written_header, key))


@pytest.mark.parallel(nprocs=4)
def test_link_write(all_paths, tmp_path):
    collection = open_linked_files(*all_paths)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**13)
    length = len(collection.properties)
    length = 8 if length > 8 else length
    comm = mpi4py.MPI.COMM_WORLD
    output_path = tmp_path / "random_linked.hdf5"
    output_path = comm.bcast(output_path, root=0)

    collection = collection.take(length, at="random")
    written_data = defaultdict(list)

    for i, (properties, particles) in enumerate(collection.objects()):
        for key, ds in particles.items():
            written_data[properties["fof_halo_tag"]].append((key, len(ds)))

    oc.write(output_path, collection)

    read_data = defaultdict(list)
    read_ds = oc.open(output_path)
    for properties, particles in read_ds.objects():
        for key, ds in particles.items():
            read_data[properties["fof_halo_tag"]].append((key, len(ds)))

    all_read = comm.gather(read_data, root=0)
    all_written = comm.gather(written_data, root=0)
    # merge the dictionaries
    if comm.Get_rank() == 0:
        read_data = {}
        written_data = {}
        for i in range(len(all_read)):
            read_data.update(all_read[i])
            written_data.update(all_written[i])
        for key in read_data:
            assert set(read_data[key]) == set(written_data[key])

    with pytest.raises(NotImplementedError):
        oc.read(output_path)
