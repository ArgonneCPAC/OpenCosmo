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
