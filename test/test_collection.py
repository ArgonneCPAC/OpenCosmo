from __future__ import annotations

import os
import random
import shutil
from collections import defaultdict
from shutil import copy
from typing import TYPE_CHECKING

import astropy.units as u
import h5py
import numpy as np
import pytest

import opencosmo as oc
from opencosmo import StructureCollection

if TYPE_CHECKING:
    from pathlib import Path


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


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

    try:
        yield path  # type: ignore
    finally:
        # Close out storage pressure immediately after each test
        if IN_GITHUB_ACTIONS:
            shutil.rmtree(path, ignore_errors=True)
        pass


@pytest.fixture
def multi_path(snapshot_path):
    return snapshot_path / "haloproperties_multi.hdf5"


@pytest.fixture
def halo_paths(snapshot_path: Path):
    files = ["haloproperties.hdf5", "haloparticles.hdf5", "sodproperties.hdf5"]
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


def test_open_structures(halo_paths, galaxy_paths):
    c1 = oc.open(galaxy_paths)
    assert isinstance(c1, oc.StructureCollection)
    c2 = oc.open(halo_paths[0], galaxy_paths[1])
    assert isinstance(c2, oc.StructureCollection)
    c3 = oc.open(halo_paths[0], *galaxy_paths)
    assert isinstance(c3, oc.StructureCollection)
    c3 = oc.open(halo_paths[0], halo_paths[1])
    assert isinstance(c3, oc.StructureCollection)
    c3 = oc.open(halo_paths[0], halo_paths[2])
    assert isinstance(c3, oc.StructureCollection)
    c3 = oc.open(*halo_paths)
    assert isinstance(c3, oc.StructureCollection)
    c3 = oc.open(*halo_paths, galaxy_paths[1])
    assert isinstance(c3, oc.StructureCollection)
    c3 = oc.open(*halo_paths, *galaxy_paths)
    assert isinstance(c3, oc.StructureCollection)
    c3 = oc.open(halo_paths[0], halo_paths[1], *galaxy_paths)
    assert isinstance(c3, oc.StructureCollection)


def test_multi_filter(multi_path):
    collection = oc.open(multi_path)
    collection = collection.filter(oc.col("sod_halo_mass") > 0)

    for ds in collection.values():
        assert all(ds.data["sod_halo_mass"] > 0)


def test_link_particles_only(halo_paths):
    collection = oc.open(halo_paths[0], halo_paths[1])
    assert isinstance(collection, oc.StructureCollection)
    for key in collection.keys():
        assert "particles" in key or key == "halo_properties"


def test_link_profiles_only(halo_paths):
    collection = oc.open(halo_paths[0], halo_paths[2])
    assert isinstance(collection, oc.StructureCollection)
    assert set(collection.keys()) == {"halo_properties", "halo_profiles"}


def test_galaxy_alias_fails_for_halos(halo_paths):
    ds = oc.open(halo_paths)
    with pytest.raises(AttributeError):
        for gal in ds.galaxies():
            pass


def test_halo_alias_fails_for_galaxies(galaxy_paths):
    ds = oc.open(galaxy_paths)
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


def test_multi_filter_write(multi_path, per_test_dir):
    collection = oc.open(multi_path)
    collection = collection.filter(oc.col("sod_halo_mass") > 0)
    for ds in collection.values():
        assert all(ds.data["sod_halo_mass"] > 0)
    oc.write(per_test_dir / "filtered.hdf5", collection)

    collection = oc.open(per_test_dir / "filtered.hdf5")
    for ds in collection.values():
        assert all(ds.select("sod_halo_mass").data > 0)


def test_select_nested_structures(halo_paths, galaxy_paths):
    collection = (
        oc.open(*halo_paths, *galaxy_paths)
        .filter(oc.col("fof_halo_mass") > 1e14)
        .take(10)
    )
    collection = collection.select(
        halo_properties=[
            "fof_halo_center_x",
            "fof_halo_center_y",
            "fof_halo_center_z",
        ],
        dm_particles=["x", "y", "z"],
        galaxies={
            "galaxy_properties": ["gal_mass_bar", "gal_mass_star"],
            "star_particles": ["x", "y", "z"],
        },
    )
    for halo in collection.halos():
        assert set(halo["halo_properties"].keys()) == {
            "fof_halo_center_x",
            "fof_halo_center_y",
            "fof_halo_center_z",
        }
        assert set(halo["dm_particles"].columns) == {"x", "y", "z"}
        assert set(halo["galaxies"]["galaxy_properties"].columns) == {
            "gal_mass_bar",
            "gal_mass_star",
        }
        assert set(halo["galaxies"]["star_particles"].columns) == {"x", "y", "z"}


def test_visit_single(halo_paths):
    collection = oc.open(*halo_paths).take(100)
    spec = {
        "dm_particles": ["x", "y", "z"],
        "halo_properties": [
            "fof_halo_center_x",
            "fof_halo_center_y",
            "fof_halo_center_z",
        ],
    }

    from time import time

    def offset(halo_properties, dm_particles):
        time()
        dx = np.mean(dm_particles["x"]) - halo_properties["fof_halo_center_x"]
        dy = np.mean(dm_particles["y"]) - halo_properties["fof_halo_center_y"]
        dz = np.mean(dm_particles["z"]) - halo_properties["fof_halo_center_z"]
        time()
        res = np.linalg.norm([dx.value, dy.value, dz.value])
        return res

    collection = collection.evaluate(offset, **spec, insert=True)
    data = collection["halo_properties"].select("offset").data
    assert not np.any(data == 0)


def test_visit_single_dataset(halo_paths):
    collection = oc.open(*halo_paths).take(200)
    spec = {
        "dm_particles": ["x", "y", "z"],
    }

    def offset(dm_particles):
        x = np.mean(dm_particles["x"])
        y = np.mean(dm_particles["y"])
        z = np.mean(dm_particles["z"])
        return {"particle_center_x": x, "particle_center_y": y, "particle_center_z": z}

    collection = collection.evaluate(offset, **spec, insert=True)
    data = (
        collection["halo_properties"]
        .select(("particle_center_x", "particle_center_y", "particle_center_z"))
        .get_data()
    )
    assert not np.any(data == 0)


def test_visit_nested_structures(halo_paths, galaxy_paths):
    collection = oc.open(*halo_paths, *galaxy_paths).take(100)
    spec = {
        "halo_properties": [
            "fof_halo_center_x",
            "fof_halo_center_y",
            "fof_halo_center_z",
        ],
        "dm_particles": ["x", "y", "z"],
        "galaxies": {
            "galaxy_properties": ["gal_mass_bar", "gal_mass_star"],
            "star_particles": ["x", "y", "z"],
        },
    }

    def offset(halo_properties, dm_particles, galaxies):
        assert set(galaxies["galaxy_properties"].columns) == {
            "gal_mass_bar",
            "gal_mass_star",
        }

        assert set(galaxies["star_particles"].columns) == {
            "x",
            "y",
            "z",
        }
        return 6

    collection = collection.evaluate(offset, **spec)
    assert np.all(collection["halo_properties"].select("offset").get_data("numpy") == 6)


def test_visit_with_return_none(halo_paths):
    collection = oc.open(*halo_paths).take(200)
    spec = {
        "dm_particles": ["x", "y", "z"],
        "halo_properties": [
            "fof_halo_center_x",
            "fof_halo_center_y",
            "fof_halo_center_z",
        ],
    }

    def offset(halo_properties, dm_particles):
        return None

    with pytest.raises(ValueError):
        collection.evaluate(offset, **spec, insert=True)


def test_visit_multiple_with_numpy(halo_paths):
    collection = oc.open(*halo_paths).take(200)
    spec = {
        "dm_particles": ["x", "y", "z"],
        "halo_properties": [
            "fof_halo_center_x",
            "fof_halo_center_y",
            "fof_halo_center_z",
            "sod_halo_com_x",
            "sod_halo_com_y",
            "sod_halo_com_z",
        ],
    }

    def offset(halo_properties, dm_particles):
        dx_fof = np.mean(dm_particles["x"]) - halo_properties["fof_halo_center_x"]
        dy_fof = np.mean(dm_particles["y"]) - halo_properties["fof_halo_center_y"]
        dz_fof = np.mean(dm_particles["z"]) - halo_properties["fof_halo_center_z"]
        dx_sod = np.mean(dm_particles["x"]) - halo_properties["sod_halo_com_x"]
        dy_sod = np.mean(dm_particles["y"]) - halo_properties["sod_halo_com_y"]
        dz_sod = np.mean(dm_particles["z"]) - halo_properties["sod_halo_com_z"]
        dr_fof = np.linalg.norm([dx_fof, dy_fof, dz_fof])
        dr_sod = np.linalg.norm([dx_sod, dy_sod, dz_sod])
        return {"dr_fof": dr_fof, "dr_sod": dr_sod}

    collection = collection.evaluate(offset, **spec, format="numpy", insert=True)
    data = collection["halo_properties"].select(["dr_fof", "dr_sod"]).get_data("numpy")
    for vals in data.values():
        assert not np.any(vals == 0)


def test_visit_multiple_with_default(halo_paths):
    collection = oc.open(*halo_paths).take(200)
    spec = {
        "dm_particles": ["x", "y", "z"],
        "halo_properties": [
            "fof_halo_center_x",
            "fof_halo_center_y",
            "fof_halo_center_z",
            "sod_halo_com_x",
            "sod_halo_com_y",
            "sod_halo_com_z",
        ],
    }

    def offset(halo_properties, dm_particles, random=10):
        dx_fof = np.mean(dm_particles["x"]) - halo_properties["fof_halo_center_x"]
        dy_fof = np.mean(dm_particles["y"]) - halo_properties["fof_halo_center_y"]
        dz_fof = np.mean(dm_particles["z"]) - halo_properties["fof_halo_center_z"]
        dx_sod = np.mean(dm_particles["x"]) - halo_properties["sod_halo_com_x"]
        dy_sod = np.mean(dm_particles["y"]) - halo_properties["sod_halo_com_y"]
        dz_sod = np.mean(dm_particles["z"]) - halo_properties["sod_halo_com_z"]
        dr_fof = np.linalg.norm([dx_fof, dy_fof, dz_fof])
        dr_sod = np.linalg.norm([dx_sod, dy_sod, dz_sod])
        return {"dr_fof": dr_fof, "dr_sod": dr_sod, "random": random}

    collection = collection.evaluate(offset, **spec, format="numpy", insert=True)
    data = (
        collection["halo_properties"]
        .select(["dr_fof", "dr_sod", "random"])
        .get_data("numpy")
    )
    for vals in data.values():
        assert not np.any(vals == 0)

    assert np.all(data["random"] == 10)


def test_visit_multiple_with_kwargs(halo_paths):
    collection = oc.open(*halo_paths).take(200)
    spec = {
        "dm_particles": ["x", "y", "z"],
        "halo_properties": [
            "fof_halo_center_x",
            "fof_halo_center_y",
            "fof_halo_center_z",
            "sod_halo_com_x",
            "sod_halo_com_y",
            "sod_halo_com_z",
        ],
    }

    def offset(halo_properties, dm_particles, random_value, other_value):
        dx_fof = np.mean(dm_particles["x"]) - halo_properties["fof_halo_center_x"]
        dy_fof = np.mean(dm_particles["x"]) - halo_properties["fof_halo_center_x"]
        dz_fof = np.mean(dm_particles["x"]) - halo_properties["fof_halo_center_x"]
        dx_sod = np.mean(dm_particles["x"]) - halo_properties["sod_halo_com_x"]
        dy_sod = np.mean(dm_particles["x"]) - halo_properties["sod_halo_com_y"]
        dz_sod = np.mean(dm_particles["x"]) - halo_properties["sod_halo_com_z"]
        _ = np.linalg.norm([dx_fof.value, dy_fof.value, dz_fof.value])
        _ = np.linalg.norm([dx_sod.value, dy_sod.value, dz_sod.value])
        return {
            "dr_fof": random_value * other_value,
            "dr_sod": random_value * other_value,
        }

    random_values = np.random.randint(0, 100, len(collection))
    other_value = 15
    collection = collection.evaluate(
        offset, **spec, insert=True, random_value=random_values, other_value=other_value
    )
    data = collection["halo_properties"].select(["dr_fof", "dr_sod"]).get_data("numpy")
    for vals in data.values():
        assert np.all(vals == random_values * other_value)


def test_visit_multiple_noinsert(halo_paths):
    collection = oc.open(*halo_paths).take(200)
    spec = {
        "dm_particles": ["x", "y", "z"],
        "halo_properties": [
            "fof_halo_center_x",
            "fof_halo_center_y",
            "fof_halo_center_z",
            "sod_halo_com_x",
            "sod_halo_com_y",
            "sod_halo_com_z",
        ],
    }

    def offset(halo_properties, dm_particles):
        dx_fof = np.mean(dm_particles["x"]) - halo_properties["fof_halo_center_x"]
        dy_fof = np.mean(dm_particles["x"]) - halo_properties["fof_halo_center_x"]
        dz_fof = np.mean(dm_particles["x"]) - halo_properties["fof_halo_center_x"]
        dx_sod = np.mean(dm_particles["x"]) - halo_properties["sod_halo_com_x"]
        dy_sod = np.mean(dm_particles["x"]) - halo_properties["sod_halo_com_y"]
        dz_sod = np.mean(dm_particles["x"]) - halo_properties["sod_halo_com_z"]
        dr_fof = np.linalg.norm([dx_fof.value, dy_fof.value, dz_fof.value])
        dr_sod = np.linalg.norm([dx_sod.value, dy_sod.value, dz_sod.value])
        return {"dr_fof": dr_fof, "dr_sod": dr_sod}

    result = collection.evaluate(offset, insert=False, **spec)
    for vals in result.values():
        assert not np.any(vals == 0)


def test_data_gets_all_particles(halo_paths):
    collection = oc.open(*halo_paths)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**14).take(
        10, at="random"
    )

    collection["dm_particles"].select("fof_halo_tag").get_data()
    for i, halo in enumerate(collection.halos()):
        for name, particle_species in halo.items():
            if "particle" not in name:
                continue
            halo_tag = halo["halo_properties"]["fof_halo_tag"]
            tag_filter = oc.col("fof_halo_tag") == halo_tag
            ds = collection[name].filter(tag_filter)
            assert np.all(
                particle_species.select("fof_halo_tag").get_data() == halo_tag
            )
            assert len(ds) == len(particle_species)


def test_visit_dataset_in_structure_collection(halo_paths):
    collection = oc.open(*halo_paths).take(20)

    def particle_id(x, y, z):
        return np.arange(len(x))

    collection = collection.evaluate(particle_id, dataset="dm_particles", insert=True)
    for halo in collection.halos(["dm_particles"]):
        particle_id = halo["dm_particles"].select("particle_id").get_data()
        assert np.all(particle_id == np.arange(len(particle_id)))


def test_visit_source(halo_paths):
    collection = oc.open(*halo_paths).take(20)

    def eval_fn(fof_halo_mass, fof_halo_tag):
        return 5

    collection = collection.evaluate(eval_fn, dataset="halo_properties")
    assert np.all(
        collection["halo_properties"].select("eval_fn").get_data(output="numpy") == 5
    )


def test_visit_dataset_in_structure_collection_nochunk(halo_paths):
    collection = oc.open(*halo_paths)

    def offset(
        fof_halo_center_x,
        fof_halo_center_y,
        fof_halo_center_z,
        sod_halo_com_x,
        sod_halo_com_y,
        sod_halo_com_z,
        sod_halo_radius,
    ):
        dx = fof_halo_center_x - sod_halo_com_x
        dy = fof_halo_center_x - sod_halo_com_x
        dz = fof_halo_center_x - sod_halo_com_x
        dr = np.sqrt(dx**2 + dy**2 + dz**2)
        return dr / sod_halo_radius

    collection_vec = collection.evaluate_on_dataset(
        offset, dataset="halo_properties", vectorize=True, insert=True
    )
    collection_loop = collection.evaluate_on_dataset(
        offset, dataset="halo_properties", insert=True
    )

    offset_vec = collection_vec["halo_properties"].select("offset").get_data("numpy")
    offset_loop = collection_loop["halo_properties"].select("offset").get_data("numpy")
    assert np.all(offset_vec == offset_loop)


def test_visit_galaxies_in_halo_collection(halo_paths, galaxy_paths):
    collection = oc.open(*halo_paths, *galaxy_paths).take(10)

    def offset(galaxy_properties, star_particles):
        total_mass = np.sum(star_particles["mass"])
        x_com = (
            star_particles["mass"]
            * star_particles["x"]
            / (total_mass * len(star_particles))
        ).sum()
        y_com = np.sum(star_particles["mass"] * star_particles["y"]) / (
            total_mass * len(star_particles)
        )
        z_com = np.sum(star_particles["mass"] * star_particles["z"]) / (
            total_mass * len(star_particles)
        )
        dx = x_com - galaxy_properties["gal_center_x"]
        dy = y_com - galaxy_properties["gal_center_y"]
        dz = z_com - galaxy_properties["gal_center_z"]
        dr = np.linalg.norm([dx.value, dy.value, dz.value])
        return dr

    collection = collection.evaluate(
        offset,
        dataset="galaxies",
        galaxy_properties=["gal_center_x", "gal_center_y", "gal_center_z"],
        star_particles=["x", "y", "z", "mass"],
        insert=True,
    )

    offsets = (
        collection["galaxies"]["galaxy_properties"].select("offset").get_data("numpy")
    )
    assert np.all(offsets > 0)


def test_data_linking(halo_paths):
    collection = oc.open(*halo_paths)
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
                assert species_halo_tags.pop() == halo_properties["fof_halo_tag"]
                n_particles += 1
            except TypeError:
                species_halo_tags = set([particle_species.select("fof_halo_tag").data])
                halo_tags.update(species_halo_tags)
                assert species_halo_tags.pop() == halo_properties["fof_halo_tag"]
                n_particles += 1
            except ValueError:
                bin_tags = set(particle_species.select("unique_tag").data)
                assert len(bin_tags) == 1
                assert bin_tags.pop() == halo_properties["fof_halo_tag"]
                n_profiles += 1

    assert n_particles > 0
    assert n_profiles > 0


def test_data_linking_with_unit_conversion(halo_paths):
    collection = oc.open(*halo_paths)
    collection = collection.with_units(
        "physical", dm_particles={"x": u.lyr, "y": u.lyr, "z": u.lyr}
    )
    collection = collection.filter(oc.col("sod_halo_mass") > 10**13).take(
        10, at="random"
    )
    for halo in collection.halos():
        dm_particle_locations = halo["dm_particles"].select(["x", "y", "z"]).get_data()
        for column in dm_particle_locations.itercols():
            assert column.unit == u.lyr


def test_unit_conversion_get_dataset(halo_paths):
    collection = oc.open(*halo_paths)
    collection = collection.with_units("unitless")
    data = collection["halo_properties"].get_data()
    for column in data.columns:
        assert data[column].unit is None
    data = collection["agn_particles"].get_data()
    for column in data.columns:
        assert data[column].unit is None


def test_data_linking_with_complex_unit_conversion(halo_paths):
    collection = oc.open(*halo_paths)
    collection = collection.with_units(
        "physical",
        conversions={u.Mpc: u.km},
        dm_particles={"x": u.lyr, "y": u.lyr, "z": u.lyr},
    )
    collection = collection.filter(oc.col("sod_halo_mass") > 10**13).take(
        2, at="random"
    )
    for halo in collection.halos():
        for key, vals in halo.items():
            if key == "dm_particles":
                dm_particle_locations = vals.select(["x", "y", "z"]).get_data()
                for column in dm_particle_locations.itercols():
                    assert column.unit == u.lyr
                continue
            if key == "halo_properties":
                val_iter = vals.values()
            else:
                val_iter = vals.get_data().values()
            found = False
            for value in val_iter:
                if isinstance(value, u.Quantity):
                    assert value.unit != u.Mpc
                    if value.unit == u.km:
                        found = True
            assert found


def test_data_linking_with_complex_unit_conversion_reset(halo_paths):
    collection = oc.open(*halo_paths)
    collection = collection.with_units(
        "physical",
        conversions={u.Mpc: u.km},
        dm_particles={"x": u.lyr, "y": u.lyr, "z": u.lyr},
    )
    collection = collection.filter(oc.col("sod_halo_mass") > 10**13).take(
        2, at="random"
    )
    for halo in collection.halos():
        for key, vals in halo.items():
            if key == "dm_particles":
                dm_particle_locations = vals.select(["x", "y", "z"]).get_data()
                for column in dm_particle_locations.itercols():
                    assert column.unit == u.lyr
                continue
            if key == "halo_properties":
                val_iter = vals.values()
            else:
                val_iter = vals.get_data().values()
            found = False
            for value in val_iter:
                if isinstance(value, u.Quantity):
                    assert value.unit != u.Mpc
                    if value.unit == u.km:
                        found = True
            assert found

    for halo in collection.with_units().halos():
        for key, vals in halo.items():
            if key == "halo_properties":
                val_iter = vals.values()
            else:
                val_iter = vals.get_data().values()
            for value in val_iter:
                if isinstance(value, u.Quantity):
                    if isinstance(value.unit, u.DexUnit):
                        continue
                    if (
                        value.unit.bases == 1
                        and value.unit.bases[0] == u.m
                        and value.unit.powers[0] == 1
                    ):
                        assert value.unit == u.Mpc


def test_data_linking_bound(halo_paths):
    collection = oc.open(*halo_paths)
    p1 = tuple(random.uniform(10, 20) for _ in range(3))
    p2 = tuple(random.uniform(60, 70) for _ in range(3))
    region = oc.make_box(p1, p2)
    collection = collection.bound(region)

    for halo in collection.objects():
        properties = halo["halo_properties"]
        for i, dim in enumerate(["x", "y", "z"]):
            val = properties[f"fof_halo_center_{dim}"].value
            assert val <= p2[i]
            assert val >= p1[i]


def test_data_link_repr(halo_paths):
    collection = oc.open(halo_paths)
    assert isinstance(collection.__repr__(), str)


def test_data_link_sort(halo_paths):
    collection = oc.open(halo_paths)
    collection = collection.sort_by("fof_halo_mass")
    fof_halo_mass = (
        collection["halo_properties"].select("fof_halo_mass").get_data("numpy")
    )
    assert np.all(fof_halo_mass[1:] >= fof_halo_mass[:-1])
    collection = collection.take(20, at="start").with_datasets(
        ["halo_properties", "dm_particles"]
    )
    for i, halo in enumerate(collection.halos()):
        assert halo["halo_properties"]["fof_halo_mass"].value == fof_halo_mass[i]
        fof_halo_tags = halo["dm_particles"].select("fof_halo_tag").get_data("numpy")
        assert np.all(fof_halo_tags == halo["halo_properties"]["fof_halo_tag"])


def test_data_link_selection(halo_paths):
    collection = oc.open(*halo_paths)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**13).take(
        10, at="random"
    )
    collection = collection.select(
        dm_particles=["x", "y", "z"], halo_properties=["fof_halo_tag", "sod_halo_mass"]
    )
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


def test_data_link_drop(halo_paths):
    collection = oc.open(*halo_paths)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**13).take(
        10, at="random"
    )
    collection = collection.drop(
        dm_particles=["x", "y", "z"], halo_properties=["fof_halo_tag", "sod_halo_mass"]
    )
    found_dm_particles = False
    for halo in collection.objects():
        properties = halo["halo_properties"]
        assert not set(properties.keys()).intersection(
            {"fof_halo_tag", "sod_halo_mass"}
        )

        if halo["dm_particles"] is not None:
            dm_particles = halo["dm_particles"]
            found_dm_particles = True
            assert not set(dm_particles.columns).intersection({"x", "y", "z"})
    assert found_dm_particles


def test_link_halos_to_galaxies(halo_paths, galaxy_paths):
    galaxy_path = galaxy_paths[0]
    collection = oc.open(*halo_paths, galaxy_path)
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
                tags = set(p.select("fof_halo_bin_tag").data)
                assert len(tags) == 1
                assert tags.pop() == fof_tag


def test_galaxy_linking(galaxy_paths):
    collection = oc.open(*galaxy_paths)
    collection = collection.filter(oc.col("gal_mass") < 10**12).take(10, at="random")
    for galaxy in collection.galaxies():
        properties = galaxy["galaxy_properties"]
        gal_tag = properties["gal_tag"]
        particle_gal_tags = set(galaxy["star_particles"].select("gal_tag").data)
        assert len(particle_gal_tags) == 1
        assert particle_gal_tags.pop() == gal_tag


def test_halo_linking_allow_empty(halo_paths):
    ds1 = oc.open(halo_paths, ignore_empty=True)
    ds2 = oc.open(halo_paths, ignore_empty=False)
    assert len(ds1) < len(ds2)
    found_halos = 0
    found_particles = 0

    for halo in ds1.halos():
        assert len(halo["dm_particles"]) > 0
        found_halos += 1
        found_particles += len(halo["dm_particles"])
    assert found_halos == len(ds1)
    assert found_particles == len(ds1["dm_particles"])


def test_halo_linking_with_empties(halo_paths):
    ds1 = oc.open(halo_paths, ignore_empty=False)
    found_profiles = False
    found_particles = False
    ds1 = ds1.filter(oc.col("fof_halo_mass") > 1e13).take(50)

    for halo in ds1.halos():
        halo_properties = halo.pop("halo_properties")
        fof_tag = halo_properties["fof_halo_tag"]
        for p in halo.values():
            try:
                tags = p.select("fof_halo_tag").get_data(unpack=False)
                assert np.all(tags == fof_tag)
                found_particles = True

            except ValueError:
                tags = p.select("fof_halo_bin_tag").get_data(unpack=False)
                assert np.all(tags == fof_tag)
                found_profiles = True

    assert found_particles and found_profiles


def test_link_write(halo_paths, per_test_dir):
    collection = oc.open(*halo_paths)
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

    oc.write(per_test_dir / "linked.hdf5", collection)
    written_data = oc.open(per_test_dir / "linked.hdf5")
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
                halo_tags.update(linked_dataset.select("fof_halo_bin_tag").data)
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


def test_simulation_collection_units(multi_path):
    collection = oc.open(multi_path)
    collection = collection.with_units(
        "physical",
        fof_halo_center_x=u.lyr,
        fof_halo_center_y=u.lyr,
        fof_halo_center_z=u.lyr,
    )
    for ds in collection.values():
        data = ds.select(
            ("fof_halo_center_x", "fof_halo_center_y", "fof_halo_center_z")
        ).get_data()
        for column in data.itercols():
            assert column.unit == u.lyr


def test_simulation_collection_order(multi_path):
    collection = oc.open(multi_path)
    for ds in collection.values():
        halo_mass = ds.select("fof_halo_mass").get_data("numpy")
        with pytest.raises(AssertionError):
            assert np.all(halo_mass[1:] <= halo_mass[:-1])

    collection = collection.sort_by("fof_halo_mass")
    for ds in collection.values():
        halo_mass = ds.select("fof_halo_mass").get_data("numpy")
        assert np.all(halo_mass[1:] >= halo_mass[:-1])


def test_simulation_collection_evaluate(multi_path):
    collection = oc.open(multi_path)

    def fof_px(fof_halo_mass, fof_halo_com_vx):
        return fof_halo_mass * fof_halo_com_vx

    collection = collection.evaluate(fof_px, vectorize=True, insert=True)
    for ds in collection.values():
        assert "fof_px" in ds.columns
        data = ds.select(["fof_halo_mass", "fof_halo_com_vx", "fof_px"]).get_data(
            "numpy"
        )
        assert np.all(data["fof_px"] == data["fof_halo_mass"] * data["fof_halo_com_vx"])


def test_simulation_collection_evaluate_noinsert(multi_path):
    collection = oc.open(multi_path)

    def fof_px(fof_halo_mass, fof_halo_com_vx):
        return fof_halo_mass * fof_halo_com_vx

    output = collection.evaluate(fof_px, vectorize=True, insert=False, format="numpy")
    for ds_name, ds in collection.items():
        assert "fof_px" not in ds.columns
        data = ds.select(["fof_halo_mass", "fof_halo_com_vx"]).get_data("numpy")
        assert np.all(
            output[ds_name]["fof_px"] == data["fof_halo_mass"] * data["fof_halo_com_vx"]
        )


def test_simulation_collection_evaluate_map_kwarg(multi_path):
    collection = oc.open(multi_path)

    def fof_px(fof_halo_mass, fof_halo_com_vx, random_value, other_value):
        return fof_halo_mass * fof_halo_com_vx * random_value / other_value

    random_data = {
        key: np.random.randint(0, 10, len(ds)) for key, ds in collection.items()
    }
    random_val = {key: np.random.randint(1, 100, 1) for key in collection.keys()}

    output = collection.evaluate(
        fof_px,
        vectorize=True,
        insert=False,
        format="numpy",
        random_value=random_data,
        other_value=random_val,
    )
    for ds_name, ds in collection.items():
        assert "fof_px" not in ds.columns
        data = ds.select(["fof_halo_mass", "fof_halo_com_vx"]).get_data("numpy")
        assert np.all(
            output[ds_name]["fof_px"]
            == data["fof_halo_mass"]
            * data["fof_halo_com_vx"]
            * random_data[ds_name]
            / random_val[ds_name]
        )


def test_simulation_collection_add(multi_path):
    collection = oc.open(multi_path)
    ds_name = next(iter(collection.keys()))
    data = np.random.randint(0, 100, len(collection[ds_name]))
    collection = collection.with_new_columns(datasets=ds_name, random_data=data)
    stored_data = collection[ds_name].select("random_data").get_data("numpy")
    assert np.all(stored_data == data)


def test_simulation_collection_add_with_descriptions(multi_path):
    collection = oc.open(multi_path)
    random_data = {
        key: np.random.randint(0, 100, len(ds)) for key, ds in collection.items()
    }
    descriptions = {
        "fof_halo_px": "x component of momentum",
        "random_data": "random data",
    }
    collection = collection.with_new_columns(
        fof_halo_px=oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx"),
        random_data=random_data,
        descriptions=descriptions,
    )
    for ds in collection.values():
        for name, desc in descriptions.items():
            assert ds.descriptions[name] == desc


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


def test_multiple_properties(galaxy_paths, halo_paths):
    galaxy_path = galaxy_paths[0]
    ds = oc.open(galaxy_path, *halo_paths)
    assert isinstance(ds, StructureCollection)


def test_chain_link(galaxy_paths, halo_paths):
    ds = oc.open(*galaxy_paths, *halo_paths)
    ds = ds.filter(oc.col("fof_halo_mass") > 1e14).take(10)
    for halo in ds.halos():
        properties = halo.pop("halo_properties")
        halo_tag = properties["fof_halo_tag"]
        for pds in halo.values():
            try:
                tags = set(pds.select("fof_halo_tag").data)
            except (ValueError, AttributeError, TypeError):
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


def test_chain_link_write(galaxy_paths, halo_paths, per_test_dir):
    ds = oc.open(*galaxy_paths, *halo_paths)
    ds = ds.filter(oc.col("fof_halo_mass") > 1e14).take(10)
    expected_types = {}
    for halo in ds.objects():
        properties = halo.pop("halo_properties")
        halo_tag = properties["fof_halo_tag"]
        types = set(halo.keys())
        expected_types[halo_tag] = types

    oc.write(per_test_dir / "linked.hdf5", ds)
    ds = oc.open(per_test_dir / "linked.hdf5")

    for halo in ds.objects():
        properties = halo.pop("halo_properties")
        halo_tag = properties["fof_halo_tag"]
        types = set(halo.keys())
        assert types == expected_types[halo_tag]
        for pds in halo.values():
            try:
                tags = set(pds.select("fof_halo_tag").data)
            except (ValueError, AttributeError, TypeError):
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


def test_data_link_sort_write(halo_paths, per_test_dir):
    collection = oc.open(halo_paths)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**14).sort_by(
        "fof_halo_mass"
    )
    oc.write(per_test_dir / "temp.hdf5", collection)
    new_collection = oc.open(per_test_dir / "temp.hdf5").take(10)
    assert np.all(
        collection["halo_properties"].select("sod_halo_mass").get_data("numpy") > 10**14
    )
    for halo in new_collection.objects(("halo_profiles",)):
        assert np.all(
            halo["halo_properties"]["fof_halo_tag"]
            == halo["halo_profiles"].select("fof_halo_bin_tag").get_data("numpy")[0]
        )


def test_add_structure_collection_with_descriptions(halo_paths):
    ds = oc.open(*halo_paths)
    ds = ds.with_new_columns(
        "dm_particles",
        gpe=oc.col("mass") * oc.col("phi"),
        descriptions="Gravitational potential energy",
    )
    ds = ds.with_new_columns(
        "halo_properties",
        com_px=oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx"),
        descriptions="x component of linear momentum",
    )
    ds = ds.filter(oc.col("fof_halo_mass") > 1e14)
    assert ds["dm_particles"].descriptions["gpe"] == "Gravitational potential energy"
    assert (
        ds["halo_properties"].descriptions["com_px"] == "x component of linear momentum"
    )


def test_data_cached_after_objects(halo_paths):
    ds = oc.open(*halo_paths)
    ds = ds.with_new_columns(
        "dm_particles",
        gpe=oc.col("mass") * oc.col("phi"),
        descriptions="Gravitational potential energy",
    )
    ds = ds.filter(oc.col("fof_halo_mass") > 1e14).take(20)
    for _ in ds.objects():
        pass

    dataset = ds["dm_particles"]
    cache = dataset._Dataset__state._DatasetState__cache
    data = cache.get_columns(("gpe",))
    assert data.get("gpe") is not None
    assert dataset.descriptions["gpe"] != "None"
