import numpy as np
import pytest

import opencosmo as oc


@pytest.fixture
def halos_600_path(lightcone_path):
    properties = lightcone_path / "step_600" / "haloproperties.hdf5"
    particles = lightcone_path / "step_600" / "haloparticles.hdf5"
    profiles = lightcone_path / "step_600" / "haloprofiles.hdf5"
    return [properties, particles, profiles]


@pytest.fixture
def galaxies_600_path(lightcone_path):
    properties = lightcone_path / "step_600" / "galaxyproperties.hdf5"
    particles = lightcone_path / "step_600" / "galaxyparticles.hdf5"
    return [properties, particles]


@pytest.fixture
def halos_601_path(lightcone_path):
    properties = lightcone_path / "step_601" / "haloproperties.hdf5"
    particles = lightcone_path / "step_601" / "haloparticles.hdf5"
    profiles = lightcone_path / "step_601" / "haloprofiles.hdf5"
    return [properties, particles, profiles]


@pytest.fixture
def galaxies_601_path(lightcone_path):
    properties = lightcone_path / "step_601" / "galaxyproperties.hdf5"
    particles = lightcone_path / "step_601" / "galaxyparticles.hdf5"
    return [properties, particles]


def verify_halo(halo):
    gravity_particle_tags = (
        halo["dm_particles"].select("fof_halo_tag").get_data("numpy")
    )
    assert np.all(gravity_particle_tags == halo["halo_properties"]["fof_halo_tag"])
    halo_bin_tags = halo["halo_profiles"].select("fof_halo_bin_tag").get_data("numpy")
    assert np.all(halo_bin_tags == halo["halo_properties"]["fof_halo_tag"])
    if "galaxy" not in halo:
        return
    for galaxy in halo["galaxies"].galaxies():
        assert (
            galaxy["galaxy_properties"]["fof_halo_tag"]
            == halo["halo_properties"]["fof_halo_tag"]
        )

        if "star_particles" not in galaxy:
            continue
        tags = galaxy["star_particles"].select("gal_tag").get_data("numpy")
        assert np.all(tags == galaxy["galaxy_properties"]["gal_tag"])


def test_open_lightcone_structure(halos_600_path, halos_601_path):
    ds = oc.open(*halos_600_path, *halos_601_path)
    for halo in ds.filter(oc.col("sod_halo_mass") > 1e14).take(10).halos():
        gravity_particle_tags = (
            halo["dm_particles"].select("fof_halo_tag").get_data("numpy")
        )
        assert np.all(gravity_particle_tags == halo["halo_properties"]["fof_halo_tag"])
        halo_bin_tags = (
            halo["halo_profiles"].select("fof_halo_bin_tag").get_data("numpy")
        )
        assert np.all(halo_bin_tags == halo["halo_properties"]["fof_halo_tag"])


def test_open_lightcone_galaxy_structure_collection(
    galaxies_600_path,
    galaxies_601_path,
):
    ds = oc.open(*galaxies_600_path, *galaxies_601_path)
    for galaxy in ds.take(100).galaxies():
        if "star_particles" not in galaxy:
            continue
        tags = galaxy["star_particles"].select("gal_tag").get_data("numpy")
        assert np.all(tags == galaxy["galaxy_properties"]["gal_tag"])


def test_open_lightcone_structure_with_galaxies(
    halos_600_path, galaxies_600_path, halos_601_path, galaxies_601_path
):
    ds = oc.open(
        *halos_600_path, *galaxies_600_path, *halos_601_path, *galaxies_601_path
    )
    ds = ds.filter(oc.col("sod_halo_mass") > 1e14).take(10)

    for halo in ds.filter(oc.col("sod_halo_mass") > 1e14).take(10).halos():
        gravity_particle_tags = (
            halo["dm_particles"].select("fof_halo_tag").get_data("numpy")
        )
        assert np.all(gravity_particle_tags == halo["halo_properties"]["fof_halo_tag"])
        halo_bin_tags = (
            halo["halo_profiles"].select("fof_halo_bin_tag").get_data("numpy")
        )
        assert np.all(halo_bin_tags == halo["halo_properties"]["fof_halo_tag"])
        for galaxy in halo["galaxies"].galaxies():
            assert (
                galaxy["galaxy_properties"]["fof_halo_tag"]
                == halo["halo_properties"]["fof_halo_tag"]
            )

            if "star_particles" not in galaxy:
                continue
            tags = galaxy["star_particles"].select("gal_tag").get_data("numpy")
            assert np.all(tags == galaxy["galaxy_properties"]["gal_tag"])


def test_write_lightcone_structure(halos_600_path, halos_601_path, tmp_path):
    ds = (
        oc.open(*halos_600_path, *halos_601_path)
        .filter(oc.col("fof_halo_mass") > 1e14)
        .take(1000)
    )
    halo_tags_start = set()
    halo_tags_end = set()
    for halo in ds.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="start").halos():
        halo_tags_start.add(halo["halo_properties"]["fof_halo_tag"])
        verify_halo(halo)

    for halo in ds.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="end").halos():
        halo_tags_end.add(halo["halo_properties"]["fof_halo_tag"])
        verify_halo(halo)
    oc.write(tmp_path / "halos.hdf5", ds)
    ds_new = oc.open(tmp_path / "halos.hdf5")

    for halo in (
        ds_new.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="start").halos()
    ):
        assert halo["halo_properties"]["fof_halo_tag"] in halo_tags_start
        verify_halo(halo)
    for halo in (
        ds_new.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="end").halos()
    ):
        assert halo["halo_properties"]["fof_halo_tag"] in halo_tags_end
        verify_halo(halo)


def test_data_link_sort_write_lightcone(halos_600_path, halos_601_path, tmp_path):
    collection = oc.open(*halos_600_path, *halos_601_path)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**14).sort_by(
        "fof_halo_mass"
    )
    output = tmp_path / "halos.hdf5"
    oc.write(output, collection)
    new_collection = oc.open(output).take(10)
    assert np.all(
        collection["halo_properties"].select("sod_halo_mass").get_data("numpy") > 10**14
    )
    for halo in new_collection.objects(("halo_profiles",)):
        assert np.all(
            halo["halo_properties"]["fof_halo_tag"]
            == halo["halo_profiles"].select("fof_halo_bin_tag").get_data("numpy")[0]
        )
