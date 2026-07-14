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
        n_checked = 0
        for structure in subset.halos():
            verify_structure_links(structure)
            n_checked += 1
        assert n_checked > 0
    else:
        # Galaxy-only structure collection
        n_checked = 0
        for galaxy in collection.take(50).galaxies():
            if "star_particles" not in galaxy:
                continue
            tags = galaxy["star_particles"].select("gal_tag").get_data("numpy")
            assert np.all(tags == galaxy["galaxy_properties"]["gal_tag"])
            n_checked += 1
        assert n_checked > 0


COMBINATION_PARAMS = [
    pytest.param(c, k, id=name) for name, (c, k) in LIGHTCONE_COMBINATIONS.items()
]


@pytest.mark.parametrize("components,expected_keys", COMBINATION_PARAMS)
def test_open_lightcone_structure_combinations(
    lightcone_files, components, expected_keys
):
    paths = [p for component in components for p in lightcone_files[component]]
    collection = oc.open(*paths)

    assert isinstance(collection, oc.StructureCollection)
    assert set(collection.keys()) == expected_keys

    verify_collection_links(collection)


@pytest.mark.parametrize("components,expected_keys", COMBINATION_PARAMS)
def test_write_lightcone_structure_combinations(
    lightcone_files, components, expected_keys, tmp_path
):
    paths = [p for component in components for p in lightcone_files[component]]
    collection = oc.open(*paths)
    # Reduce to a manageable subset before writing (the common usage pattern).
    if "halo_properties" in collection.keys():
        collection = collection.filter(oc.col("fof_halo_mass") > 1e14).take(1000)
    else:
        collection = collection.take(1000)

    output = tmp_path / "collection.hdf5"
    oc.write(output, collection)
    reopened = oc.open(output)

    assert isinstance(reopened, oc.StructureCollection)
    assert set(reopened.keys()) == expected_keys

    verify_collection_links(reopened)


def test_evaluate_into_galaxy_properties(halos_600_path, galaxies_600_path):
    def offset(halo_properties, galaxy_properties):
        dx = galaxy_properties["gal_com_x"] - halo_properties["fof_halo_center_x"]
        dy = galaxy_properties["gal_com_y"] - halo_properties["fof_halo_center_y"]
        dz = galaxy_properties["gal_com_z"] - halo_properties["fof_halo_center_z"]
        dist2 = dx**2 + dy**2 + dz**2
        dist = np.sqrt(dist2)
        offset = dist / halo_properties["sod_halo_RVir"]  # divide by virial radius
        return {"gal_offset": offset}

    ds = oc.open(*halos_600_path, galaxies_600_path[0])
    ds = ds.evaluate(
        offset,
        dataset="galaxy_properties",
        insert=True,
        format="numpy",
        halo_properties=["fof_halo_center_*", "sod_halo_RVir"],
        galaxy_properties=["gal_com_*"],
    )
    for halo in ds.halos():
        _ = halo["galaxy_properties"].select("gal_offset").get_data()


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


def test_write_lightcone_structure_with_galaxies(
    halos_600_path, halos_601_path, galaxies_600_path, galaxies_601_path, tmp_path
):
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


def test_redshift_bound(halos_600_path, halos_601_path, tmp_path):
    collection = oc.open(*halos_600_path, *halos_601_path)
    collection = collection.with_redshift_range(0.038, 0.039)

    collection = collection.filter(oc.col("sod_halo_mass") > 10**14)
    for halo in collection.halos():
        redshift = halo["halo_properties"]["redshift"]
        assert redshift > 0.038 and redshift < 0.039
        verify_halo(halo)
