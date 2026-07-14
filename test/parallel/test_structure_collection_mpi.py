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


@pytest.fixture
def lightcone_files(lightcone_path):
    """Map a component name to the per-step files that provide it."""

    def files(stem):
        return [
            lightcone_path / step / f"{stem}.hdf5" for step in ("step_600", "step_601")
        ]

    return {
        "halo_properties": files("haloproperties"),
        "halo_particles": files("haloparticles"),
        "halo_profiles": files("haloprofiles"),
        "galaxy_properties": files("galaxyproperties"),
        "galaxy_particles": files("galaxyparticles"),
    }


# Each entry is the set of components combined into a lightcone structure
# collection and the dataset keys we expect the resulting collection to expose.
LIGHTCONE_COMBINATIONS = {
    "halo_particles": (
        ["halo_properties", "halo_particles"],
        {
            "agn_particles",
            "dm_particles",
            "gas_particles",
            "star_particles",
            "halo_properties",
        },
    ),
    "halo_profiles": (
        ["halo_properties", "halo_profiles"],
        {"halo_profiles", "halo_properties"},
    ),
    "halo_particles_profiles": (
        ["halo_properties", "halo_particles", "halo_profiles"],
        {
            "agn_particles",
            "dm_particles",
            "gas_particles",
            "star_particles",
            "halo_profiles",
            "halo_properties",
        },
    ),
    "halo_particles_profiles_galaxy_properties": (
        ["halo_properties", "halo_particles", "halo_profiles", "galaxy_properties"],
        {
            "agn_particles",
            "dm_particles",
            "gas_particles",
            "star_particles",
            "halo_profiles",
            "galaxy_properties",
            "halo_properties",
        },
    ),
    "halo_galaxy_properties": (
        ["halo_properties", "galaxy_properties"],
        {"galaxy_properties", "halo_properties"},
    ),
    "halo_galaxy_properties_particles": (
        ["halo_properties", "galaxy_properties", "galaxy_particles"],
        {"galaxies", "halo_properties"},
    ),
    "galaxy_properties_particles": (
        ["galaxy_properties", "galaxy_particles"],
        {"star_particles", "galaxy_properties"},
    ),
}

COMBINATION_PARAMS = [
    pytest.param(c, k, id=name) for name, (c, k) in LIGHTCONE_COMBINATIONS.items()
]


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


def verify_structure_links(structure):
    """Verify every linked dataset in a structure points back to its host."""
    host_tag = structure["halo_properties"]["fof_halo_tag"]
    for name, linked in structure.items():
        if name == "halo_properties":
            continue
        if name == "halo_profiles":
            tags = linked.select("fof_halo_bin_tag").get_data("numpy")
            assert np.all(tags == host_tag)
        elif name == "galaxy_properties":
            tags = linked.select("fof_halo_tag").get_data("numpy")
            assert np.all(tags == host_tag)
        elif name == "galaxies":
            for galaxy in linked.galaxies():
                assert galaxy["galaxy_properties"]["fof_halo_tag"] == host_tag
        elif "particles" in name:
            tags = linked.select("fof_halo_tag").get_data("numpy")
            assert np.all(tags == host_tag)


def verify_collection_links(collection, n=10):
    """Verify links across a structure collection of halos or galaxies."""
    if "halo_properties" in collection.keys():
        subset = collection.filter(oc.col("sod_halo_mass") > 1e13).take(n)
        for structure in subset.halos():
            verify_structure_links(structure)
    else:
        for galaxy in collection.take(50).galaxies():
            if "star_particles" not in galaxy:
                continue
            tags = galaxy["star_particles"].select("gal_tag").get_data("numpy")
            assert np.all(tags == galaxy["galaxy_properties"]["gal_tag"])


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("components,expected_keys", COMBINATION_PARAMS)
def test_open_lightcone_structure_combinations(
    lightcone_files, components, expected_keys
):
    paths = [p for component in components for p in lightcone_files[component]]
    collection = oc.open(*paths)

    assert isinstance(collection, oc.StructureCollection)
    assert set(collection.keys()) == expected_keys

    verify_collection_links(collection)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("components,expected_keys", COMBINATION_PARAMS)
def test_write_lightcone_structure_combinations(
    lightcone_files, components, expected_keys, per_test_dir
):
    paths = [p for component in components for p in lightcone_files[component]]
    collection = oc.open(*paths)
    # Reduce to a manageable subset before writing (the common usage pattern).
    if "halo_properties" in collection.keys():
        collection = collection.filter(oc.col("fof_halo_mass") > 1e14).take(1000)
    else:
        collection = collection.take(1000)

    output = per_test_dir / "collection.hdf5"
    oc.write(output, collection)
    reopened = oc.open(output)

    assert isinstance(reopened, oc.StructureCollection)
    assert set(reopened.keys()) == expected_keys

    verify_collection_links(reopened)


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
