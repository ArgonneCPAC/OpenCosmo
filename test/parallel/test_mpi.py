import random
from collections import defaultdict
from pathlib import Path

import h5py
import mpi4py
import numpy as np
import pytest
from pytest_mpi.parallel_assert import parallel_assert

import opencosmo as oc
from opencosmo import open_linked_files
from opencosmo.collection import SimulationCollection


@pytest.fixture
def input_path(snapshot_path):
    return snapshot_path / "haloproperties.hdf5"


@pytest.fixture
def particle_path(snapshot_path):
    return snapshot_path / "haloparticles.hdf5"


@pytest.fixture
def malformed_header_path(input_path, tmp_path):
    update = {"n_dm": "foo"}
    return update_simulation_parameter(input_path, update, tmp_path, "malformed_header")


@pytest.fixture
def galaxy_paths(snapshot_path: Path):
    files = ["galaxyproperties.hdf5", "galaxyparticles.hdf5"]
    hdf_files = [snapshot_path / file for file in files]
    return list(hdf_files)


@pytest.fixture
def galaxy_paths_2(snapshot_path: Path):
    files = ["galaxyproperties2.hdf5", "galaxyparticles2.hdf5"]
    hdf_files = [snapshot_path / file for file in files]
    return list(hdf_files)


@pytest.fixture
def all_paths(snapshot_path: Path):
    files = ["haloparticles.hdf5", "haloproperties.hdf5", "sodproperties.hdf5"]

    hdf_files = [snapshot_path / file for file in files]
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
def test_collect(input_path):
    with oc.open(input_path) as f:
        ds = f.filter(oc.col("sod_halo_mass") > 0).take(100, at="random").collect()

    parallel_assert(len(ds.data) == 400)


@pytest.mark.parallel(nprocs=4)
def test_filter_write(input_path, tmp_path):
    comm = mpi4py.MPI.COMM_WORLD
    temporary_path = tmp_path / "filtered.hdf5"
    temporary_path = comm.bcast(temporary_path, root=0)

    ds = oc.open(input_path)
    ds = ds.filter(oc.col("sod_halo_mass") > 0)

    oc.write(temporary_path, ds)
    data = ds.data
    ds.close()

    ds = oc.open(temporary_path)
    written_data = ds.data
    for column in ds.columns:
        parallel_assert(np.all(data[column] == written_data[column]))

    parallel_assert(all(data == written_data))


@pytest.mark.parallel(nprocs=4)
def test_filter_zerolength(input_path, tmp_path):
    comm = mpi4py.MPI.COMM_WORLD
    temporary_path = tmp_path / "filtered.hdf5"
    temporary_path = comm.bcast(temporary_path, root=0)

    ds = oc.open(input_path)
    ds = ds.filter(oc.col("sod_halo_mass") > 0)
    rank = comm.Get_rank()
    if rank == 1:
        ds = ds.filter(oc.col("sod_halo_mass") > 1e20)

    oc.write(temporary_path, ds)
    data = ds.data
    ds.close()

    ds = oc.open(temporary_path)
    written_data = ds.data

    if rank == 1:
        assert len(written_data) == 0
    else:
        for column in ds.columns:
            assert np.all(data[column] == written_data[column])

    assert all(data == written_data)


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
def test_link_read(all_paths):
    collection = open_linked_files(*all_paths)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**13)
    length = len(collection.properties)
    length = 8 if length > 8 else length
    collection = collection.take(8, "random")
    for i, (properties, particles) in enumerate(collection.objects()):
        halo_tag = properties["fof_halo_tag"]
        for species, ds in particles.items():
            data = ds.data
            if len(data) == 0:
                continue
            if species == "halo_profiles":
                assert len(data) == 1
                assert data["fof_halo_bin_tag"][0][0] == halo_tag
                continue
            halo_tags = np.unique(ds.data["fof_halo_tag"])
            assert len(halo_tags) == 1
            assert halo_tags[0] == halo_tag


@pytest.mark.parallel(nprocs=4)
def test_link_write(all_paths, tmp_path):
    collection = open_linked_files(*all_paths)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**13.5)
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


@pytest.mark.parallel(nprocs=4)
def test_box_query_collect(input_path):
    ds = oc.open(input_path)
    center = tuple(random.uniform(30, 60) for _ in range(3))
    width = tuple(random.uniform(10, 20) for _ in range(3))

    center = mpi4py.MPI.COMM_WORLD.bcast(center)
    width = mpi4py.MPI.COMM_WORLD.bcast(width)

    reg1 = oc.Box(center, width)
    original_data = ds.data
    ds = ds.bound(reg1)
    ds = ds.collect()
    data = ds.data
    for i, dim in enumerate(["x", "y", "z"]):
        name = f"fof_halo_center_{dim}"
        min_ = center[i] - width[i] / 2
        max_ = center[i] + width[i] / 2
        original_col = original_data[name]
        mask = (original_col < max_) & (original_col > min_)
        original_data = original_data[mask]

        col = data[name]
        min = col.min()
        max = col.max()
        parallel_assert(min >= min_ and np.isclose(min, min_, 0.1))
        parallel_assert(max <= max_ and np.isclose(max, max_, 0.1))

    length = mpi4py.MPI.COMM_WORLD.bcast(len(ds))
    parallel_assert(len(ds) == length)


@pytest.mark.parallel(nprocs=4)
def test_box_query_chain(input_path):
    ds = oc.open(input_path).with_units("scalefree")
    center1 = (30, 40, 50)
    width1 = (10, 15, 20)
    reg1 = oc.Box(center1, width1)

    center2 = (31, 41, 51)
    width2 = (5, 7.5, 10)
    reg2 = oc.Box(center2, width2)

    ds = ds.bound(reg1)
    ds = ds.bound(reg2)

    ds = ds.collect()
    data = ds.data
    for i, dim in enumerate(["x", "y", "z"]):
        col = data[f"fof_halo_center_{dim}"]
        min_ = center2[i] - width2[i] / 2
        max_ = center2[i] + width2[i] / 2
        min = col.min()
        max = col.max()
        parallel_assert(min >= min_ and np.isclose(min, min_, 0.1))
        parallel_assert(max <= max_ and np.isclose(max, max_, 0.1))

        parallel_assert(max <= max_ and np.isclose(max, max_, 0.1))

    size = mpi4py.MPI.COMM_WORLD.bcast(len(ds))
    parallel_assert(len(ds) == size)


@pytest.mark.parallel(nprocs=4)
def test_collection_of_linked(galaxy_paths, galaxy_paths_2, tmp_path):
    galaxies_1 = open_linked_files(*galaxy_paths)
    galaxies_2 = open_linked_files(*galaxy_paths_2)
    datasets = {"scidac_01": galaxies_1, "scidac_02": galaxies_2}
    tmp_path = tmp_path / "galaxies.hdf5"
    tmp_path = mpi4py.MPI.COMM_WORLD.bcast(tmp_path, root=0)

    collection = SimulationCollection(datasets)
    oc.write(tmp_path, collection)

    dataset = oc.open(tmp_path)
    dataset = dataset.filter(oc.col("gal_mass") > 10**12).take(10, at="random")
    for ds in dataset.values():
        for props, particles in ds.objects():
            gal_tag = props["gal_tag"]
            gal_tags = set(particles.data["gal_tag"])
            assert len(gal_tags) == 1
            assert gal_tags.pop() == gal_tag
