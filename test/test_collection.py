from pathlib import Path
from typing import defaultdict

import pytest

import opencosmo as oc
from opencosmo.collection import SimulationCollection
from opencosmo.link import open_linked_files
from opencosmo.link.collection import StructureCollection


@pytest.fixture
def multi_path(data_path):
    return data_path / "haloproperties_multi.hdf5"


@pytest.fixture
def halo_paths(data_path: Path):
    files = ["haloparticles.hdf5", "haloproperties.hdf5", "sodproperties.hdf5"]
    hdf_files = [data_path / file for file in files]
    return list(hdf_files)


@pytest.fixture
def galaxy_paths(data_path: Path):
    files = ["galaxyproperties.hdf5", "galaxyparticles.hdf5"]
    hdf_files = [data_path / file for file in files]
    return list(hdf_files)


@pytest.fixture
def galaxy_paths_2(data_path: Path):
    files = ["galaxyproperties2.hdf5", "galaxyparticles2.hdf5"]
    hdf_files = [data_path / file for file in files]
    return list(hdf_files)


def test_multi_filter(multi_path):
    collection = oc.read(multi_path)
    collection = collection.filter(oc.col("sod_halo_mass") > 0)

    for ds in collection.values():
        assert all(ds.data["sod_halo_mass"] > 0)


def test_multi_repr(multi_path):
    collection = oc.read(multi_path)
    assert isinstance(collection.__repr__(), str)


def test_multi_filter_write(multi_path, tmp_path):
    collection = oc.read(multi_path)
    collection = collection.filter(oc.col("sod_halo_mass") > 0)
    for ds in collection.values():
        assert all(ds.data["sod_halo_mass"] > 0)
    oc.write(tmp_path / "filtered.hdf5", collection)

    collection = oc.read(tmp_path / "filtered.hdf5")
    for ds in collection.values():
        assert all(ds.data["sod_halo_mass"] > 0)


def test_data_linking(halo_paths):
    collection = open_linked_files(*halo_paths)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**13).take(
        10, at="random"
    )
    particle_species = filter(lambda name: "particles" in name, collection.keys())
    n_particles = 0
    n_profiles = 0
    for properties, particles in collection.objects():
        halo_tags = set()
        for name, particle_species in particles.items():
            if len(particle_species) == 0:
                continue
            try:
                halo_tags = set(particle_species.data["fof_halo_tag"])
                assert len(halo_tags) == 1
                halo_tags.update(particle_species.data["fof_halo_tag"])
                n_particles += 1
            except KeyError:
                bin_tags = [tag for tag in particle_species.data["unique_tag"][0]]
                bin_tags = set(bin_tags)
                assert len(bin_tags) == 1
                halo_tags.update(bin_tags)
                n_profiles += 1

        assert len(set(halo_tags)) == 1
        assert halo_tags.pop() == properties["fof_halo_tag"]
    assert n_particles > 0
    assert n_profiles > 0


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
    for properties, particles in collection.objects():
        assert set(properties.keys()) == {"fof_halo_tag", "sod_halo_mass"}
        if particles["dm_particles"] is not None:
            dm_particles = particles["dm_particles"]
            found_dm_particles = True
            assert set(dm_particles.data.colnames) == {"x", "y", "z"}
    assert found_dm_particles


def test_link_halos_to_galaxies(halo_paths, galaxy_paths):
    galaxy_path = galaxy_paths[0]
    collection = open_linked_files(*halo_paths, galaxy_path)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**14).take(10)
    for properties, particles in collection.objects():
        fof_tag = properties["fof_halo_tag"]
        for p in particles.values():
            try:
                tags = set(p.data["fof_halo_tag"])
                assert len(tags) == 1
                assert tags.pop() == fof_tag
            except KeyError:
                tags = set(p.data["fof_halo_bin_tag"][0])
                assert len(tags) == 1
                assert tags.pop() == fof_tag


def test_galaxy_linking(galaxy_paths):
    collection = open_linked_files(*galaxy_paths)
    collection = collection.filter(oc.col("gal_mass") < 10**12).take(10, at="random")
    for properties, star_particles in collection.objects():
        gal_tag = properties["gal_tag"]
        particle_gal_tags = set(star_particles.data["gal_tag"])
        assert len(particle_gal_tags) == 1
        assert particle_gal_tags.pop() == gal_tag


def test_link_write(halo_paths, tmp_path):
    collection = open_linked_files(*halo_paths)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**13.5).take(
        10, at="random"
    )
    original_output = defaultdict(list)
    for properties, particles in collection.objects():
        for name, particle_species in particles.items():
            if particle_species is None:
                continue
            original_output[properties["fof_halo_tag"]].append(name)

    read_output = defaultdict(list)
    oc.write(tmp_path / "linked.hdf5", collection)
    written_data = oc.open(tmp_path / "linked.hdf5")
    n = 0
    for properties, particles in written_data.objects():
        halo_tags = set()
        n += 1
        for linked_type, linked_dataset in particles.items():
            if linked_dataset is None:
                continue
            read_output[properties["fof_halo_tag"]].append(linked_type)

            if "particles" not in linked_type:
                bin_tags = [tag for tag in linked_dataset.data["fof_halo_bin_tag"][0]]
                halo_tags.update(bin_tags)
            else:
                species_tags = set(linked_dataset.data["fof_halo_tag"])
                halo_tags.update(species_tags)

        assert len(halo_tags) == 1
        assert halo_tags.pop() == properties["fof_halo_tag"]

    for key in original_output.keys():
        assert set(original_output[key]) == set(read_output[key])

    with pytest.raises(NotImplementedError):
        oc.read(tmp_path / "linked.hdf5")

    assert n == 10


def test_simulation_collection_broadcast_attribute(multi_path):
    collection = oc.read(multi_path)
    for key, value in collection.redshift.items():
        assert isinstance(key, str)
        assert isinstance(value, float)


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
        for props, particles in ds.objects():
            gal_tag = props["gal_tag"]
            gal_tags = set(particles.data["gal_tag"])
            assert len(gal_tags) == 1
            assert gal_tags.pop() == gal_tag
            j += 1

    assert j == 100


def test_multiple_properties(galaxy_paths, halo_paths):
    galaxy_path = galaxy_paths[0]
    ds = open_linked_files(galaxy_path, *halo_paths)
    assert isinstance(ds, StructureCollection)
