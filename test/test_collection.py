from pathlib import Path

import pytest

import opencosmo as oc
from opencosmo.link import open_linked_files
from typing import defaultdict


@pytest.fixture
def multi_path(data_path):
    return data_path / "haloproperties_multi.hdf5"


@pytest.fixture
def all_paths(data_path: Path):
    files = ["haloparticles.hdf5", "haloproperties.hdf5", "sodproperties.hdf5"]

    hdf_files = [data_path / file for file in files]
    return list(hdf_files)


def test_multi_filter(multi_path):
    collection = oc.read(multi_path)
    collection = collection.filter(oc.col("sod_halo_mass") > 0)

    for ds in collection.datasets():
        assert all(ds.data["sod_halo_mass"] > 0)


def test_multi_filter_write(multi_path, tmp_path):
    collection = oc.read(multi_path)
    collection = collection.filter(oc.col("sod_halo_mass") > 0)
    for ds in collection.datasets():
        assert all(ds.data["sod_halo_mass"] > 0)
    oc.write(tmp_path / "filtered.hdf5", collection)

    collection = oc.read(tmp_path / "filtered.hdf5")
    for ds in collection.datasets():
        assert all(ds.data["sod_halo_mass"] > 0)


def test_data_linking(all_paths):
    collection = open_linked_files(*all_paths)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**14).take(
        10, at="random"
    )
    particle_species = filter(lambda name: "particles" in name, collection.keys())
    n_particles = 0
    n_profiles = 0
    for properties, particles in collection.objects():
        halo_tags = set()
        for name, particle_species in particles.items():
            if particle_species is None:
                continue
            try:
                halo_tags.update(particle_species.data["fof_halo_tag"])
                n_particles += 1
            except KeyError:
                bin_tags = [tag for tag in particle_species.data["unique_tag"][0]]
                halo_tags.update(bin_tags)
                n_profiles += 1
        assert len(set(halo_tags)) == 1
        assert halo_tags.pop() == properties["fof_halo_tag"]
    assert n_particles > 0
    assert n_profiles > 0


def test_link_write(all_paths, tmp_path):
    collection = open_linked_files(*all_paths)
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
