import random
from collections import defaultdict
from pathlib import Path
from shutil import copy

import h5py
import numpy as np
import pytest

import opencosmo as oc
from opencosmo import StructureCollection, open_linked_files
from opencosmo.collection import SimulationCollection


@pytest.fixture
def multi_path(snapshot_path):
    return snapshot_path / "haloproperties_multi.hdf5"


@pytest.fixture
def halo_paths(snapshot_path: Path):
    files = ["haloparticles.hdf5", "haloproperties.hdf5", "sodproperties.hdf5"]
    hdf_files = [snapshot_path / file for file in files]
    return list(hdf_files)


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
def conditional_path(multi_path, tmp_path):
    path = tmp_path / "conditional_load.hdf5"
    copy(multi_path, path)
    with h5py.File(path, "a") as f:
        f["scidac1"].create_group("load/if")
        f["scidac1/load/if"].attrs["foo"] = True
    return path


def test_multi_filter(multi_path):
    collection = oc.open(multi_path)
    collection = collection.filter(oc.col("sod_halo_mass") > 0)

    for ds in collection.values():
        assert all(ds.data["sod_halo_mass"] > 0)


def test_galaxy_alias_fails_for_halos(halo_paths):
    ds = open_linked_files(halo_paths)
    with pytest.raises(AttributeError):
        for gal in ds.galaxies():
            pass


def test_halo_alias_fails_for_galaxies(galaxy_paths):
    ds = open_linked_files(galaxy_paths)
    with pytest.raises(AttributeError):
        for gal in ds.halos():
            pass


def test_multi_repr(multi_path):
    collection = oc.open(multi_path)
    assert isinstance(collection.__repr__(), str)


def test_conditional_load(conditional_path):
    ds = oc.open(conditional_path)
    assert isinstance(ds, oc.Dataset)
    ds = oc.open(conditional_path, foo=True)
    assert isinstance(ds, oc.SimulationCollection)


def test_multi_filter_write(multi_path, tmp_path):
    collection = oc.open(multi_path)
    collection = collection.filter(oc.col("sod_halo_mass") > 0)
    for ds in collection.values():
        assert all(ds.data["sod_halo_mass"] > 0)
    oc.write(tmp_path / "filtered.hdf5", collection)

    collection = oc.open(tmp_path / "filtered.hdf5")
    for ds in collection.values():
        assert all(ds.select("sod_halo_mass").data > 0)


def test_data_linking(halo_paths):
    collection = open_linked_files(*halo_paths)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**13).take(
        10, at="random"
    )
    particle_species = filter(lambda name: "particles" in name, collection.keys())
    n_particles = 0
    n_profiles = 0
    for halo in collection.halos():
        halo_properties = halo.pop("halo_properties")
        halo_tags = set()
        for name, particle_species in halo.items():
            if len(particle_species) == 0:
                continue
            try:
                species_halo_tags = set(particle_species.select("fof_halo_tag").data)
                assert len(species_halo_tags) == 1
                halo_tags.update(species_halo_tags)
                n_particles += 1
            except ValueError:
                bin_tags = [
                    tag for tag in particle_species.select("unique_tag").data[0]
                ]
                bin_tags = set(bin_tags)
                assert len(bin_tags) == 1
                halo_tags.update(bin_tags)
                n_profiles += 1

        assert len(set(halo_tags)) == 1
        assert halo_tags.pop() == halo_properties["fof_halo_tag"]
    assert n_particles > 0
    assert n_profiles > 0


def test_data_linking_bound(halo_paths):
    collection = open_linked_files(*halo_paths)
    p1 = tuple(random.uniform(30, 40) for _ in range(3))
    p2 = tuple(random.uniform(50, 60) for _ in range(3))
    region = oc.make_box(p1, p2)
    collection = collection.bound(region)

    for halo in collection.objects():
        properties = halo["halo_properties"]
        for i, dim in enumerate(["x", "y", "z"]):
            val = properties[f"fof_halo_center_{dim}"].value
            assert val <= p2[i]
            assert val >= p1[i]


def test_data_link_repr(halo_paths):
    collection = open_linked_files(halo_paths)
    assert isinstance(collection.__repr__(), str)


def test_data_link_selection(halo_paths):
    collection = open_linked_files(*halo_paths)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**13).take(
        10, at="random"
    )
    collection = collection.select(["x", "y", "z"], dataset="dm_particles")
    collection = collection.select(["fof_halo_tag", "sod_halo_mass"])
    found_dm_particles = False
    for halo in collection.objects():
        properties = halo["halo_properties"]
        assert set(properties.keys()) == {"fof_halo_tag", "sod_halo_mass"}
        assert np.all(properties["sod_halo_mass"].value > 10**13)

        if halo["dm_particles"] is not None:
            dm_particles = halo["dm_particles"]
            found_dm_particles = True
            assert set(dm_particles.columns) == {"x", "y", "z"}
    assert found_dm_particles


def test_link_halos_to_galaxies(halo_paths, galaxy_paths):
    galaxy_path = galaxy_paths[0]
    collection = open_linked_files(*halo_paths, galaxy_path)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**14).take(10)
    for halo in collection.halos():
        properties = halo.pop("halo_properties")
        fof_tag = properties["fof_halo_tag"]
        for p in halo.values():
            try:
                tags = set(p.select("fof_halo_tag").data)
                assert len(tags) == 1
                assert tags.pop() == fof_tag
            except ValueError:
                tags = set(p.select("fof_halo_bin_tag").data[0])
                assert len(tags) == 1
                assert tags.pop() == fof_tag


def test_galaxy_linking(galaxy_paths):
    collection = open_linked_files(*galaxy_paths)
    collection = collection.filter(oc.col("gal_mass") < 10**12).take(10, at="random")
    for galaxy in collection.galaxies():
        properties = galaxy["galaxy_properties"]
        gal_tag = properties["gal_tag"]
        particle_gal_tags = set(galaxy["star_particles"].select("gal_tag").data)
        assert len(particle_gal_tags) == 1
        assert particle_gal_tags.pop() == gal_tag


def test_link_write(halo_paths, tmp_path):
    collection = open_linked_files(*halo_paths)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**13.5).take(
        10, at="random"
    )
    original_output = defaultdict(list)
    for halo in collection.objects():
        properties = halo.pop("halo_properties")
        for name, particle_species in halo.items():
            if particle_species is None:
                continue
            original_output[properties["fof_halo_tag"]].append(name)

    read_output = defaultdict(list)

    oc.write(tmp_path / "linked.hdf5", collection)
    written_data = oc.open(tmp_path / "linked.hdf5")
    n = 0
    for halo in written_data.objects():
        halo_tags = set()
        n += 1
        properties = halo.pop("halo_properties")
        for linked_type, linked_dataset in halo.items():
            if linked_dataset is None:
                continue
            read_output[properties["fof_halo_tag"]].append(linked_type)

            if "particles" not in linked_type:
                bin_tags = [
                    tag for tag in linked_dataset.select("fof_halo_bin_tag").data[0]
                ]
                halo_tags.update(bin_tags)
            else:
                species_tags = set(linked_dataset.select("fof_halo_tag").data)
                halo_tags.update(species_tags)

        assert len(halo_tags) == 1
        assert halo_tags.pop() == properties["fof_halo_tag"]

    for key in original_output.keys():
        assert set(original_output[key]) == set(read_output[key])


def test_simulation_collection_derive(multi_path):
    collection = oc.open(multi_path)
    collection = collection.with_new_columns(
        fof_halo_px=oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    )
    for ds in collection.values():
        assert "fof_halo_px" in ds.columns
        assert "fof_halo_px" in ds.data.columns


def test_simulation_collection_broadcast_attribute(multi_path):
    collection = oc.open(multi_path)
    for key, value in collection.redshift.items():
        assert isinstance(key, str)
        assert isinstance(value, float)


def test_simulation_collection_bound(multi_path):
    collection = oc.open(multi_path)
    p1 = tuple(random.uniform(10, 20) for _ in range(3))
    p2 = tuple(random.uniform(30, 40) for _ in range(3))
    region = oc.make_box(p1, p2)
    collection = collection.bound(region)

    for name, properties in collection.items():
        data = properties.data
        for i, dim in enumerate(["x", "y", "z"]):
            val = data[f"fof_halo_center_{dim}"].value
            assert np.all(val <= p2[i])
            assert np.all(val >= p1[i])


def test_collection_of_linked(galaxy_paths, galaxy_paths_2, tmp_path):
    galaxies_1 = open_linked_files(*galaxy_paths)
    galaxies_2 = open_linked_files(*galaxy_paths_2)
    datasets = {"scidac_01": galaxies_1, "scidac_02": galaxies_2}

    collection = SimulationCollection(datasets)
    collection = collection.filter(oc.col("gal_mass") > 10**12).take(50, at="start")

    oc.write(tmp_path / "galaxies.hdf5", collection)

    dataset = oc.open(tmp_path / "galaxies.hdf5")
    j = 0
    for ds in dataset.values():
        for galaxy in ds.galaxies():
            properties = galaxy["galaxy_properties"]
            gal_tag = properties["gal_tag"]
            gal_tags = set(galaxy["star_particles"].select("gal_tag").data)
            assert len(gal_tags) == 1
            assert gal_tags.pop() == gal_tag
            j += 1

    assert j == 100


def test_multiple_properties(galaxy_paths, halo_paths):
    galaxy_path = galaxy_paths[0]
    ds = open_linked_files(galaxy_path, *halo_paths)
    assert isinstance(ds, StructureCollection)


def test_chain_link(galaxy_paths, halo_paths):
    ds = open_linked_files(*galaxy_paths, *halo_paths)
    ds = ds.filter(oc.col("fof_halo_mass") > 1e14).take(10)
    for halo in ds.halos():
        properties = halo.pop("halo_properties")
        halo_tag = properties["fof_halo_tag"]
        for pds in halo.values():
            try:
                tags = set(pds.select("fof_halo_tag").data)
            except (ValueError, AttributeError):
                continue

            assert len(tags) == 1
            assert tags.pop() == halo_tag
        for galaxy in halo["galaxies"].galaxies():
            gal_properties = galaxy["galaxy_properties"]
            gal_tag = gal_properties["gal_tag"]
            assert gal_properties["fof_halo_tag"] == halo_tag
            gal_tags = set(galaxy["star_particles"].select("gal_tag").data)
            assert len(gal_tags) == 1
            assert gal_tags.pop() == gal_tag


def test_chain_link_write(galaxy_paths, halo_paths, tmp_path):
    ds = open_linked_files(*galaxy_paths, *halo_paths)
    ds = ds.filter(oc.col("fof_halo_mass") > 1e14).take(10)
    oc.write(tmp_path / "linked.hdf5", ds)
    ds = oc.open(tmp_path / "linked.hdf5")
    for halo in ds.objects():
        properties = halo.pop("halo_properties")
        halo_tag = properties["fof_halo_tag"]
        for pds in halo.values():
            try:
                tags = set(pds.select("fof_halo_tag").data)
            except (ValueError, AttributeError):
                continue

            assert len(tags) == 1
            assert tags.pop() == halo_tag
        for galaxy in halo["galaxies"].objects():
            gal_properties = galaxy["galaxy_properties"]
            gal_tag = gal_properties["gal_tag"]
            assert gal_properties["fof_halo_tag"] == halo_tag
            gal_tags = set(galaxy["star_particles"].select("gal_tag").data)
            assert len(gal_tags) == 1
            assert gal_tags.pop() == gal_tag
