from __future__ import annotations

import os
from collections import defaultdict
from functools import reduce
from typing import TYPE_CHECKING

import astropy.units as u
import h5py
import mpi4py
import numpy as np
import pytest
from mpi4py import MPI
from pytest_mpi.parallel_assert import parallel_assert

import opencosmo as oc

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def multi_path(snapshot_path):
    return snapshot_path / "haloproperties_multi.hdf5"


@pytest.fixture
def input_path(snapshot_path):
    return snapshot_path / "haloproperties.hdf5"


@pytest.fixture
def particle_path(snapshot_path):
    return snapshot_path / "haloparticles.hdf5"


@pytest.fixture
def profile_path(snapshot_path):
    return snapshot_path / "sodproperties.hdf5"


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


@pytest.mark.timeout(20)
@pytest.mark.parallel(nprocs=4)
def test_mpi(input_path):
    with oc.open(input_path) as f:
        data = f.data

    parallel_assert(len(data) != 0)


@pytest.mark.parallel(nprocs=4)
def test_structure_collection_open(input_path, profile_path):
    oc.open(input_path, profile_path)


@pytest.mark.timeout(20)
@pytest.mark.parallel(nprocs=4)
def test_structure_collection_open_2(input_path, profile_path):
    comm = MPI.COMM_WORLD
    if comm.Get_rank() in [0, 2]:
        oc.open(profile_path, input_path)
    else:
        oc.open(input_path, profile_path)


@pytest.mark.timeout(20)
@pytest.mark.parallel(nprocs=4)
def test_partitioning_includes_all(input_path):
    with oc.open(input_path) as f:
        tags = f.select("fof_halo_tag").data

    comm = mpi4py.MPI.COMM_WORLD
    all_tags = comm.allgather(tags)
    all_tags = reduce(lambda left, right: left.union(set(right)), all_tags, set())

    file = h5py.File(input_path)
    original_tags = set(file["data"]["fof_halo_tag"][:])

    parallel_assert(all_tags == original_tags)


@pytest.mark.timeout(1)
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


@pytest.mark.timeout(20)
@pytest.mark.parallel(nprocs=4)
def test_filters(input_path):
    ds = oc.open(input_path)
    ds = ds.filter(oc.col("sod_halo_mass") > 0)
    data = ds.data
    ds.close()
    parallel_assert(len(data) != 0)
    parallel_assert(all(data["sod_halo_mass"] > 0))


@pytest.mark.timeout(20)
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


@pytest.mark.timeout(20)
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
        read_tags = comm.allgather([])
    else:
        read_tags = comm.allgather(data["fof_halo_tag"])

    written_tags = comm.allgather(written_data["fof_halo_tag"])

    read_tags = reduce(lambda left, right: left.union(set(right)), read_tags, set())
    written_tags = reduce(
        lambda left, right: left.union(set(right)), written_tags, set()
    )

    parallel_assert(read_tags == written_tags)


@pytest.mark.timeout(20)
@pytest.mark.parallel(nprocs=4)
def test_link_read(all_paths):
    collection = oc.open(*all_paths)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**13)
    length = len(collection["halo_properties"])
    length = 8 if length > 8 else length
    collection = collection.take(8, "random")
    for i, halo in enumerate(collection.objects()):
        halo_properties = halo.pop("halo_properties")
        halo_tag = halo_properties["fof_halo_tag"]
        for species, ds in halo.items():
            data = ds.data
            if len(data) == 0:
                continue
            if species == "halo_profiles":
                print(isinstance(data, dict))
                assert np.all(data["fof_halo_bin_tag"] == halo_tag)
                continue
            halo_tags = np.unique(ds.data["fof_halo_tag"])
            assert len(halo_tags) == 1
            assert halo_tags[0] == halo_tag


@pytest.mark.timeout(20)
@pytest.mark.parallel(nprocs=4)
def test_evaluate_structure(all_paths):
    collection = oc.open(*all_paths).take(100)

    spec = {
        "dm_particles": ["x", "y", "z"],
        "halo_properties": [
            "fof_halo_center_x",
            "fof_halo_center_y",
            "fof_halo_center_z",
        ],
    }

    def offset(halo_properties, dm_particles):
        dx = np.mean(dm_particles["x"]) - halo_properties["fof_halo_center_x"]
        dy = np.mean(dm_particles["y"]) - halo_properties["fof_halo_center_y"]
        dz = np.mean(dm_particles["z"]) - halo_properties["fof_halo_center_z"]
        return np.linalg.norm([dx, dy, dz])

    collection = collection.evaluate(offset, **spec, insert=True, format="numpy")
    data = collection["halo_properties"].select("offset").data
    assert not np.any(data == 0)


@pytest.mark.timeout(20)
@pytest.mark.parallel(nprocs=4)
def test_evaluate_structure_write(all_paths, tmp_path):
    comm = mpi4py.MPI.COMM_WORLD
    temporary_path = tmp_path / "test.hdf5"
    temporary_path = comm.bcast(temporary_path, root=0)
    collection = oc.open(*all_paths).take(100)

    spec = {
        "dm_particles": ["x", "y", "z"],
        "halo_properties": [
            "fof_halo_center_x",
            "fof_halo_center_y",
            "fof_halo_center_z",
        ],
    }

    def offset(halo_properties, dm_particles):
        dx = np.mean(dm_particles["x"]) - halo_properties["fof_halo_center_x"]
        dy = np.mean(dm_particles["y"]) - halo_properties["fof_halo_center_y"]
        dz = np.mean(dm_particles["z"]) - halo_properties["fof_halo_center_z"]
        return np.linalg.norm([dx.value, dy.value, dz.value])

    collection = collection.evaluate(offset, **spec, insert=True)
    oc.write(temporary_path, collection)
    collection = oc.open(temporary_path)
    data = collection["halo_properties"].select("offset").data
    assert not np.any(data == 0)


@pytest.mark.timeout(20)
@pytest.mark.parallel(nprocs=4)
def test_link_write(all_paths, tmp_path):
    collection = oc.open(*all_paths)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**13.5)
    length = len(collection["halo_properties"])
    length = 8 if length > 8 else length
    comm = mpi4py.MPI.COMM_WORLD
    output_path = tmp_path / "random_linked.hdf5"
    output_path = comm.bcast(output_path, root=0)

    collection = collection.take(length, at="random")
    written_data = defaultdict(list)

    lens = []
    for i, structure in enumerate(collection.objects()):
        halo_properties = structure.pop("halo_properties")
        for key, ds in structure.items():
            if key == "dm_particles":
                lens.append(len(ds))
            written_data[halo_properties["fof_halo_tag"]].append((key, len(ds)))
            try:
                tag = halo_properties["fof_halo_tag"]
                tags = set(ds.select("fof_halo_tag").data)
                parallel_assert(len(tags) == 1 and tags.pop() == tag)
            except ValueError:
                continue

    oc.write(output_path, collection)
    read_data = defaultdict(list)
    read_ds = oc.open(output_path)

    for i, structure in enumerate(read_ds.objects()):
        halo_properties = structure.pop("halo_properties")
        for key, ds in structure.items():
            read_data[halo_properties["fof_halo_tag"]].append((key, len(ds)))
            try:
                tag = halo_properties["fof_halo_tag"]
                tags = set(ds.select("fof_halo_tag").data)
                assert len(tags) == 1 and tags.pop() == tag
            except ValueError:
                continue

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


@pytest.mark.timeout(20)
@pytest.mark.parallel(nprocs=4)
def test_chain_link(all_paths, galaxy_paths, tmp_path):
    collection = oc.open(*all_paths, *galaxy_paths)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**13.5)
    length = len(collection["halo_properties"])
    length = 8 if length > 8 else length
    comm = mpi4py.MPI.COMM_WORLD
    output_path = tmp_path / "random_linked.hdf5"
    output_path = comm.bcast(output_path, root=0)

    collection = collection.take(length, at="random")
    written_data = defaultdict(list)

    for i, halos in enumerate(collection.objects()):
        properties = halos.pop("halo_properties")
        for key, ds in halos.items():
            written_data[properties["fof_halo_tag"]].append((key, len(ds)))

    oc.write(output_path, collection)

    read_data = defaultdict(list)
    read_ds = oc.open(output_path)
    for halo in read_ds.objects():
        properties = halo.pop("halo_properties")
        for key, ds in halo.items():
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


@pytest.mark.timeout(20)
@pytest.mark.parallel(nprocs=4)
def test_box_query_chain(input_path):
    ds = oc.open(input_path).with_units("scalefree")
    original_data = ds.data
    bounds = ds.region.bounds
    widths = tuple(b[1] - b[0] for b in bounds)
    p1 = tuple(b[0] + w / 4 for b, w in zip(bounds, widths))
    p2 = tuple(b[0] + 3 * w / 4 for b, w in zip(bounds, widths))
    reg1 = oc.make_box(p1, p2)

    p1 = tuple(b[0] + w / 3 for b, w in zip(bounds, widths))
    p2 = tuple(b[0] + 2 * w / 3 for b, w in zip(bounds, widths))
    reg2 = oc.make_box(p1, p2)
    ds = ds.bound(reg1)
    ds = ds.bound(reg2)

    data = ds.data
    for i, dim in enumerate(["x", "y", "z"]):
        colname = f"fof_halo_center_{dim}"
        col = data[colname]
        min_ = p1[i]
        max_ = p2[i]
        mask = (original_data[colname].value < max_) & (
            original_data[colname].value > min_
        )
        original_data = original_data[mask]
        min = col.min()
        max = col.max()
        parallel_assert(min.value >= min_)
        parallel_assert(max.value <= max_)

    parallel_assert(set(original_data["fof_halo_tag"]) == set(data["fof_halo_tag"]))


@pytest.mark.timeout(20)
@pytest.mark.parallel(nprocs=4)
def test_box_query_zerolength(input_path):
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()

    ds = oc.open(input_path)
    p1 = (3, 3, 3)
    p2 = (10, 10, 10)
    reg1 = oc.make_box(p1, p2)

    ds = ds.bound(reg1)
    if rank == 0:
        assert len(ds) > 0

    parallel_assert(len(ds) == 0, participating=rank > 0)


@pytest.mark.timeout(20)
@pytest.mark.parallel(nprocs=4)
def test_derive_multiply(input_path):
    ds = oc.open(input_path)
    derived = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    ds = ds.with_new_columns(fof_halo_px=derived)
    data = ds.data
    parallel_assert("fof_halo_px" in data.columns)
    parallel_assert(
        data["fof_halo_px"].unit
        == data["fof_halo_mass"].unit * data["fof_halo_com_vx"].unit
    )
    parallel_assert(
        np.all(
            np.isclose(
                data["fof_halo_px"].value,
                data["fof_halo_mass"].value * data["fof_halo_com_vx"].value,
            )
        )
    )


@pytest.mark.timeout(20)
@pytest.mark.parallel(nprocs=4)
def test_add_column(input_path):
    ds = oc.open(input_path)
    data = np.random.randint(0, 100, len(ds)) * u.deg
    ds = ds.with_new_columns(random_data=data)
    stored_data = ds.select("random_data").data
    assert np.all(data == stored_data)


@pytest.mark.timeout(20)
@pytest.mark.parallel(nprocs=4)
def test_add_column_write(input_path, tmp_path):
    comm = mpi4py.MPI.COMM_WORLD
    temporary_path = tmp_path / "test.hdf5"
    temporary_path = comm.bcast(temporary_path, root=0)

    ds = oc.open(input_path)
    data = np.random.randint(0, 100, len(ds)) * u.deg
    ds = ds.with_new_columns(random_data=data)
    oc.write(temporary_path, ds)
    written_data = oc.open(temporary_path).select("random_data").get_data()
    assert np.all(written_data == data)


@pytest.mark.timeout(20)
@pytest.mark.parallel(nprocs=4)
def test_evaluate(input_path):
    ds = oc.open(input_path)

    def fof_px(fof_halo_mass, fof_halo_com_vx):
        return fof_halo_mass * fof_halo_com_vx

    ds = ds.evaluate(fof_px, vectorize=True, insert=True)
    assert "fof_px" in ds.columns
    data = ds.select(["fof_halo_mass", "fof_halo_com_vx", "fof_px"]).get_data("numpy")
    assert np.all(data["fof_px"] == data["fof_halo_mass"] * data["fof_halo_com_vx"])


@pytest.mark.timeout(0.25, method="thread")
@pytest.mark.parallel(nprocs=4)
def test_evaluate_write(input_path, tmp_path):
    comm = mpi4py.MPI.COMM_WORLD
    temporary_path = tmp_path / "test.hdf5"
    temporary_path = comm.bcast(temporary_path, root=0)
    ds = oc.open(input_path)

    def fof_px(fof_halo_mass, fof_halo_com_vx):
        return fof_halo_mass * fof_halo_com_vx

    ds = ds.evaluate(fof_px, vectorize=True, insert=True)
    oc.write(temporary_path, ds)
    ds = oc.open(temporary_path)

    assert "fof_px" in ds.columns
    data = ds.select(["fof_halo_mass", "fof_halo_com_vx", "fof_px"]).get_data("numpy")
    assert np.all(data["fof_px"] == data["fof_halo_mass"] * data["fof_halo_com_vx"])


@pytest.mark.timeout(20)
@pytest.mark.parallel(nprocs=4)
def test_derive_write(input_path, tmp_path):
    comm = mpi4py.MPI.COMM_WORLD
    temporary_path = tmp_path / "derived.hdf5"
    temporary_path = comm.bcast(temporary_path, root=0)

    ds = oc.open(input_path)
    derived = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    ds = ds.with_new_columns(fof_halo_px=derived)
    oc.write(temporary_path, ds)
    original_data = ds.data
    written_data = oc.open(temporary_path).data

    parallel_assert("fof_halo_px" in written_data.columns)
    parallel_assert(
        written_data["fof_halo_px"].unit == original_data["fof_halo_px"].unit
    )
    parallel_assert(
        np.all(
            np.isclose(
                written_data["fof_halo_px"].value, original_data["fof_halo_px"].value
            )
        )
    )


@pytest.mark.timeout(20)
@pytest.mark.parallel(nprocs=4)
def test_simcollection_write(multi_path, tmp_path):
    comm = mpi4py.MPI.COMM_WORLD
    temporary_path = tmp_path / "collection.hdf5"
    temporary_path = comm.bcast(temporary_path, root=0)
    data = oc.open(multi_path)
    halo_tags = {}
    columns = {}
    for name, sim in data.items():
        columns[name] = sim.columns
        sim_tags = sim.select("fof_halo_tag").get_data("numpy")
        halo_tags[name] = sim_tags

    oc.write(temporary_path, data)
    written_data = oc.open(temporary_path)
    assert isinstance(written_data, oc.SimulationCollection)
    for name, sim in written_data.items():
        assert sim.columns == columns[name]
        sim_tags = sim.select("fof_halo_tag").get_data("numpy")


@pytest.mark.timeout(1)
@pytest.mark.parallel(nprocs=4)
def test_simcollection_write_one_missing(multi_path, tmp_path):
    comm = mpi4py.MPI.COMM_WORLD
    temporary_path = tmp_path / "collection.hdf5"
    temporary_path = comm.bcast(temporary_path, root=0)
    data = oc.open(multi_path)

    halo_tags = {}
    if comm.Get_rank() == 0:
        key_to_drop = next(iter(data.keys()))
        data.pop(key_to_drop)

    for name, sim in data.items():
        sim_tags = sim.select("fof_halo_tag").get_data("numpy")
        halo_tags[name] = sim_tags

    all_tags = comm.allgather(halo_tags)
    all_halo_tags = defaultdict(lambda: np.array([], dtype=int))
    for rank_tags in all_tags:
        for simkey, tags in rank_tags.items():
            all_halo_tags[simkey] = np.append(all_halo_tags[simkey], tags)

    oc.write(temporary_path, data)
    written_data = oc.open(temporary_path)
    assert isinstance(written_data, oc.SimulationCollection)
    assert len(written_data.keys()) == 2
    written_halo_tags = {}
    for name, sim in written_data.items():
        sim_tags = sim.select("fof_halo_tag").get_data("numpy")
        written_halo_tags[name] = sim_tags

    all_tags = comm.allgather(written_halo_tags)
    all_written_halo_tags = defaultdict(lambda: np.array([], dtype=int))
    for rank_tags in all_tags:
        for simkey, tags in rank_tags.items():
            all_written_halo_tags[simkey] = np.append(
                all_written_halo_tags[simkey], tags
            )

    for simkey, tags in all_written_halo_tags.items():
        parallel_assert(np.all(tags == all_halo_tags[simkey]))
