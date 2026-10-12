"""
Characterization tests for the user-facing semantics of StructureCollection.

These pin the mapping structure, deferred link rebuilding, and lightcone-only
behavior that must survive the move to a StructureState.
"""

import numpy as np
import pytest

import opencosmo as oc


@pytest.fixture
def halo_paths(test_data):
    return test_data.snapshot.primary.halos


@pytest.fixture
def galaxy_paths(test_data):
    return test_data.snapshot.primary.galaxies


@pytest.fixture
def lightcone_halo_paths(test_data):
    return test_data.lightcone.step(600).halos + test_data.lightcone.step(601).halos


@pytest.fixture
def lightcone_galaxy_paths(test_data):
    return (
        test_data.lightcone.step(600).galaxies + test_data.lightcone.step(601).galaxies
    )


def _halo_properties_path(paths):
    return [p for p in paths if p.name == "haloproperties.hdf5"]


def _run_values(values):
    """The value of each run of repeated values, in order."""
    if len(values) == 0:
        return values
    return values[np.r_[0, np.flatnonzero(np.diff(values)) + 1]]


def test_keys_put_source_last(halo_paths):
    collection = oc.open(*halo_paths)
    keys = collection.keys()
    assert keys[-1] == "halo_properties"
    assert keys.count("halo_properties") == 1
    assert {"dm_particles", "halo_profiles"} <= set(keys)
    assert [name for name, _ in collection.items()] == keys


def test_keys_stable_across_transformations(halo_paths):
    collection = oc.open(*halo_paths)
    keys = collection.keys()
    assert collection.filter(oc.col("fof_halo_mass") > 1e13).keys() == keys
    assert collection.sort_by("fof_halo_mass").keys() == keys
    assert collection.take(10).keys() == keys
    assert collection.select(dm_particles=["x"]).keys() == keys
    assert collection.with_units("physical").keys() == keys


def test_halo_collection_values_are_datasets(halo_paths):
    collection = oc.open(*halo_paths)
    assert all(isinstance(v, oc.Dataset) for v in collection.values())


def test_galaxy_collection_values_are_datasets(galaxy_paths):
    collection = oc.open(*galaxy_paths)
    assert collection.keys() == ["star_particles", "galaxy_properties"]
    assert all(isinstance(v, oc.Dataset) for v in collection.values())


def test_halo_galaxy_properties_collection_links_dataset(halo_paths, galaxy_paths):
    collection = oc.open(halo_paths[0], galaxy_paths[0])
    assert collection.keys() == ["galaxy_properties", "halo_properties"]
    assert all(isinstance(v, oc.Dataset) for v in collection.values())


def test_halo_galaxy_collection_nests_galaxies(halo_paths, galaxy_paths):
    collection = oc.open(*halo_paths, *galaxy_paths)
    keys = collection.keys()
    assert "galaxies" in keys
    assert "galaxy_properties" not in keys
    assert keys[-1] == "halo_properties"
    for name, value in collection.items():
        if name == "galaxies":
            assert isinstance(value, oc.StructureCollection)
            assert value.keys() == ["star_particles", "galaxy_properties"]
        else:
            assert isinstance(value, oc.Dataset)


def test_lightcone_collection_values_are_lightcones(lightcone_halo_paths):
    collection = oc.open(*lightcone_halo_paths)
    assert collection.keys()[-1] == "halo_properties"
    assert all(isinstance(v, oc.Lightcone) for v in collection.values())


def test_lightcone_collection_nests_galaxies(
    lightcone_halo_paths, lightcone_galaxy_paths
):
    collection = oc.open(
        *_halo_properties_path(lightcone_halo_paths), *lightcone_galaxy_paths
    )
    assert collection.keys() == ["galaxies", "halo_properties"]
    assert isinstance(collection["halo_properties"], oc.Lightcone)
    galaxies = collection["galaxies"]
    assert isinstance(galaxies, oc.StructureCollection)
    assert galaxies.keys() == ["star_particles", "galaxy_properties"]
    assert all(isinstance(v, oc.Lightcone) for v in galaxies.values())


def test_with_datasets_hides_source(halo_paths):
    collection = oc.open(*halo_paths).take(5, at="start")
    hidden = collection.with_datasets("dm_particles")
    assert hidden.keys() == ["dm_particles"]
    assert len(hidden) == len(collection)
    for halo in hidden.halos():
        assert set(halo.keys()) == {"dm_particles"}


def test_with_datasets_keeps_source_last(halo_paths):
    collection = oc.open(*halo_paths)
    subset = collection.with_datasets(["halo_properties", "dm_particles"])
    assert subset.keys() == ["dm_particles", "halo_properties"]


def test_with_datasets_hides_source_with_nested_galaxies(halo_paths, galaxy_paths):
    collection = oc.open(*halo_paths, *galaxy_paths)
    hidden = collection.with_datasets("galaxies")
    assert hidden.keys() == ["galaxies"]
    assert isinstance(hidden["galaxies"], oc.StructureCollection)


def test_chained_row_ops_rebuild_links(halo_paths):
    collection = (
        oc.open(*halo_paths)
        .filter(oc.col("fof_halo_mass") > 1e13)
        .sort_by("fof_halo_mass", invert=True)
        .take(10, at="start")
    )
    halo_tags = collection["halo_properties"].select("fof_halo_tag").get_data("numpy")
    particle_tags = collection["dm_particles"].select("fof_halo_tag").get_data("numpy")
    assert np.array_equal(_run_values(particle_tags), halo_tags)

    pieces = [
        halo["dm_particles"].select("fof_halo_tag").get_data("numpy")
        for halo in collection.halos(["dm_particles"])
    ]
    assert np.array_equal(np.concatenate(pieces), particle_tags)


def test_child_select_between_row_ops(halo_paths):
    collection = (
        oc.open(*halo_paths)
        .filter(oc.col("fof_halo_mass") > 1e13)
        .select(dm_particles=["fof_halo_tag", "x"])
        .filter(oc.col("sod_halo_mass") > 1e13)
        .take(10, at="start")
    )
    particles = collection["dm_particles"]
    assert set(particles.columns) == {"fof_halo_tag", "x"}

    halo_tags = collection["halo_properties"].select("fof_halo_tag").get_data("numpy")
    particle_tags = particles.select("fof_halo_tag").get_data("numpy")
    assert np.array_equal(_run_values(particle_tags), halo_tags)


@pytest.mark.parametrize("which", ["halos", "galaxies"])
def test_lightcone_only_methods_fail_on_snapshots(halo_paths, galaxy_paths, which):
    collection = oc.open(*(halo_paths if which == "halos" else galaxy_paths))
    with pytest.raises(AttributeError):
        collection.get_pixels()
    with pytest.raises(AttributeError):
        collection.cone_search((0, 0), 1)
    with pytest.raises(AttributeError):
        collection.box_search((0, 0), (1, 1))
    with pytest.raises(AttributeError):
        collection.pixel_search(np.array([0]))
    with pytest.raises(AttributeError):
        collection.with_redshift_range(0.0, 1.0)


def test_lightcone_objects_align_with_links(lightcone_halo_paths):
    collection = (
        oc.open(*lightcone_halo_paths)
        .filter(oc.col("sod_halo_mass") > 1e14)
        .take(10, at="start")
    )
    n_checked = 0
    for halo in collection.halos(["dm_particles"]):
        tags = halo["dm_particles"].select("fof_halo_tag").get_data("numpy")
        assert len(tags) > 0
        assert np.all(tags == halo["halo_properties"]["fof_halo_tag"])
        n_checked += 1
    assert n_checked == len(collection)


@pytest.mark.parametrize("invert", [False, True])
def test_sorted_lightcone_objects_align_with_links(lightcone_halo_paths, invert):
    collection = (
        oc.open(*lightcone_halo_paths)
        .filter(oc.col("sod_halo_mass") > 1e14)
        .sort_by("fof_halo_mass", invert=invert)
        .take(10, at="start")
    )
    n_checked = 0
    for halo in collection.halos(["dm_particles"]):
        tags = halo["dm_particles"].select("fof_halo_tag").get_data("numpy")
        assert len(tags) > 0
        assert np.all(tags == halo["halo_properties"]["fof_halo_tag"])
        n_checked += 1
    assert n_checked == len(collection)


@pytest.mark.xfail(
    strict=True,
    reason="Per-structure evaluate on a nested path stores the result under the "
    "dotted key instead of replacing the nested collection",
)
def test_evaluate_nested_path_replaces_nested_collection(halo_paths, galaxy_paths):
    collection = oc.open(*halo_paths, *galaxy_paths).take(10)

    def doubled_x(x):
        return 2 * x

    result = collection.evaluate(
        doubled_x, dataset="galaxies.star_particles", insert=True
    )
    assert result.keys() == collection.keys()
    assert "doubled_x" in result["galaxies"]["star_particles"].columns


def _assert_cached_after_objects(collection, dataset_name, column):
    for _ in collection.objects():
        pass
    state = collection[dataset_name]._state
    uuid = state.column_map[column]
    cached = state.cache.get_data({(uuid, column)})
    assert cached.get(uuid, {}).get(column) is not None


@pytest.mark.xfail(
    strict=True, reason="with_units drops derived_columns, so objects() skips them"
)
def test_derived_columns_survive_with_units(halo_paths):
    collection = oc.open(*halo_paths).with_new_columns(
        "dm_particles", gpe=oc.col("mass") * oc.col("phi")
    )
    collection = (
        collection.with_units("physical").filter(oc.col("fof_halo_mass") > 1e14).take(5)
    )
    _assert_cached_after_objects(collection, "dm_particles", "gpe")


@pytest.mark.xfail(
    strict=True,
    reason="Nested-path with_new_columns drops derived_columns, so objects() "
    "skips them",
)
def test_derived_columns_survive_nested_with_new_columns(halo_paths, galaxy_paths):
    collection = oc.open(*halo_paths, *galaxy_paths).with_new_columns(
        "dm_particles", gpe=oc.col("mass") * oc.col("phi")
    )
    collection = collection.with_new_columns(
        "galaxies.galaxy_properties", double_mass=2 * oc.col("gal_mass_star")
    )
    collection = collection.filter(oc.col("fof_halo_mass") > 1e14).take(5)
    _assert_cached_after_objects(collection, "dm_particles", "gpe")
