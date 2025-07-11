from collections import defaultdict
from pathlib import Path

import h5py
import mpi4py
import numpy as np
import pytest
from pytest_mpi.parallel_assert import parallel_assert

import opencosmo as oc
from opencosmo import open_linked_files


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
                parallel_assert(len(tags) == 1 and tags.pop() == tag)
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


@pytest.mark.parallel(nprocs=4)
def test_chain_link(all_paths, galaxy_paths, tmp_path):
    collection = open_linked_files(*all_paths, *galaxy_paths)
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
        mask = (original_data[colname] < max_) & (original_data[colname] > min_)
        original_data = original_data[mask]
        min = col.min()
        max = col.max()
        parallel_assert(min >= min_)
        parallel_assert(max <= max_)

    parallel_assert(set(original_data["fof_halo_tag"]) == set(data["fof_halo_tag"]))


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
            data["fof_halo_px"].value
            == data["fof_halo_mass"].value * data["fof_halo_com_vx"].value
        )
    )


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
