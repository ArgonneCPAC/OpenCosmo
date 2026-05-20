import os
import shutil

import numpy as np
import pytest
from mpi4py import MPI
from opencosmo.mpi import get_comm_world

import opencosmo as oc

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


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


@pytest.fixture
def per_test_dir(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
):
    """
    Creates a unique directory for each test and deletes it after the test finishes.

    Uses tmp_path_factory so you can control base temp location via pytest's
    tempdir handling, and also so it can be used from broader-scoped fixtures
    if needed.
    """
    # request.node.nodeid is unique across parameterizations; sanitize for filesystem
    nodeid = (
        request.node.nodeid.replace("/", "_")
        .replace("::", "__")
        .replace("[", "_")
        .replace("]", "_")
    )

    path = tmp_path_factory.mktemp(nodeid)
    comm = MPI.COMM_WORLD
    path_to_return = comm.bcast(path)

    try:
        yield path_to_return
    finally:
        # Close out storage pressure immediately after each test
        if IN_GITHUB_ACTIONS:
            shutil.rmtree(path, ignore_errors=True)


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


@pytest.mark.parallel(nprocs=4)
def test_open_lightcone_structure_with_galaxies(
    halos_600_path, galaxies_600_path, halos_601_path, galaxies_601_path
):
    ds = oc.open(
        *halos_600_path, *galaxies_600_path, *halos_601_path, *galaxies_601_path
    )
    ds = ds.filter(oc.col("sod_halo_mass") > 1e14).take(10)

    for halo in ds.filter(oc.col("sod_halo_mass") > 1e14).take(10).halos():
        verify_halo(halo)


@pytest.mark.parallel(nprocs=4)
def test_write_lightcone_structure(halos_600_path, halos_601_path, per_test_dir):
    comm = get_comm_world()
    ds = (
        oc.open(
            *halos_600_path,
            *halos_601_path,
        )
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
    all_halos_read = set(
        np.concatenate(
            comm.allgather(
                ds["halo_properties"].select("fof_halo_tag").get_data("numpy")
            )
        )
    )

    oc.write(per_test_dir / "halos.hdf5", ds)
    ds_new = oc.open(per_test_dir / "halos.hdf5")

    all_halos = set(
        np.concatenate(
            comm.allgather(
                ds_new["halo_properties"].select("fof_halo_tag").get_data("numpy")
            )
        )
    )

    for halo in (
        ds_new.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="start").halos()
    ):
        verify_halo(halo)
    for halo in (
        ds_new.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="end").halos()
    ):
        verify_halo(halo)

    assert halo_tags_start.issubset(all_halos)
    assert halo_tags_end.issubset(all_halos)
    assert all_halos_read == all_halos


@pytest.mark.parallel(nprocs=4)
def test_write_lightcone_structure_with_galaxies(
    halos_600_path, halos_601_path, galaxies_600_path, galaxies_601_path, per_test_dir
):
    comm = get_comm_world()
    ds = (
        oc.open(
            *halos_600_path, *halos_601_path, *galaxies_600_path, *galaxies_601_path
        )
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
    all_halos_read = set(
        np.concatenate(
            comm.allgather(
                ds["halo_properties"].select("fof_halo_tag").get_data("numpy")
            )
        )
    )

    oc.write(per_test_dir / "halos.hdf5", ds)
    ds_new = oc.open(per_test_dir / "halos.hdf5")

    all_halos = set(
        np.concatenate(
            comm.allgather(
                ds_new["halo_properties"].select("fof_halo_tag").get_data("numpy")
            )
        )
    )

    for halo in (
        ds_new.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="start").halos()
    ):
        verify_halo(halo)
    for halo in (
        ds_new.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="end").halos()
    ):
        verify_halo(halo)

    assert halo_tags_start.issubset(all_halos)
    assert halo_tags_end.issubset(all_halos)
    assert all_halos_read == all_halos
