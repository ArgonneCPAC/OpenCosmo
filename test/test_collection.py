from pathlib import Path

import pytest

import opencosmo as oc
from opencosmo.collection import open_linked


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
    collection = open_linked(*all_paths)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**13.5).take(
        10, at="random"
    )
    particle_species = filter(lambda name: "particles" in name, collection.keys())
    for properties, particles in collection.objects(list(particle_species)):
        halo_tags = set()
        for particle_species in particles.values():
            halo_tags.update(particle_species.data["fof_halo_tag"])
        assert len(set(halo_tags)) == 1
        assert halo_tags.pop() == properties["fof_halo_tag"]


def test_link_write(all_paths, tmp_path):
    collection = open_linked(*all_paths)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**13.5).take(
        10, at="random"
    )
    oc.write(tmp_path / "linked.hdf5", collection)
    written_data = oc.open(tmp_path / "linked.hdf5")
    n = 0
    particle_species = filter(lambda name: "particles" in name, written_data.keys())
    for properties, particles in written_data.objects(list(particle_species)):
        halo_tags = set()
        n += 1
        for particle_type, particle_species in particles.items():
            species_tags = set(particle_species.data["fof_halo_tag"])
            halo_tags.update(species_tags)

        assert len(halo_tags) == 1
        assert halo_tags.pop() == properties["fof_halo_tag"]

    with pytest.raises(NotImplementedError):
        oc.read(tmp_path / "linked.hdf5")

    assert n == 10
