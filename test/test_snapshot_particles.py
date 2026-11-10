from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.cosmology.units as cu
import astropy.units as u
import h5py
import numpy as np
import opencosmo as oc
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def counts(snapshot_particle_writer) -> dict[str, int]:
    return dict(snapshot_particle_writer.counts)


@pytest.fixture
def snapshot_particles_path(tmp_path, snapshot_particle_writer) -> Path:
    return snapshot_particle_writer.write(tmp_path / "snapshotparticles.hdf5")


def _read(path: Path, name: str, field: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f[f"{name}/data/{field}"][:]


def _fields(path: Path, name: str) -> set[str]:
    with h5py.File(path, "r") as f:
        return set(f[f"{name}/data"].keys())


def test_open_returns_particle_collection(snapshot_particles_path, counts):
    collection = oc.open(snapshot_particles_path)
    assert isinstance(collection, oc.ParticleCollection)
    assert sorted(collection.keys()) == sorted(counts)
    assert collection.dtype == "snapshot_particles"
    assert collection.header.file.data_type == "snapshot_particles"
    for name, dataset in collection.items():
        assert isinstance(dataset, oc.Dataset)
        assert len(dataset) == counts[name]
        assert set(dataset.columns) == _fields(snapshot_particles_path, name)
        assert np.array_equal(
            dataset.select("id").get_data("numpy"),
            _read(snapshot_particles_path, name, "id"),
        )


def test_missing_particle_type(snapshot_particles_path):
    collection = oc.open(snapshot_particles_path)
    with pytest.raises(KeyError, match="halo_particles"):
        collection["halo_particles"]


def test_units(snapshot_particles_path, snapshot_particle_writer):
    h = snapshot_particle_writer.h
    collection = oc.open(snapshot_particles_path)
    gas = collection["gas_particles"]
    data = gas.select("x", "wind_velocity", "sfr", "wind_decoupling_time").get_data()
    assert data["x"].unit == u.Mpc
    assert data["wind_velocity"].unit == u.km / u.s
    assert data["sfr"].unit == u.Msun / u.yr
    assert data["wind_decoupling_time"].unit == u.Gyr
    raw_x = _read(snapshot_particles_path, "gas_particles", "x")
    assert np.allclose(data["x"].value, raw_x / h)

    scalefree = collection.with_units("scalefree")["gas_particles"]
    assert scalefree.select("x").get_data().unit == u.Mpc / cu.littleh

    physical = collection.with_units("physical")["dm_particles"]
    x = physical.select("x").get_data()
    raw_x = _read(snapshot_particles_path, "dm_particles", "x")
    a = 1 / (1 + snapshot_particle_writer.redshift)
    assert np.allclose(x.value, raw_x / h * a)


def test_with_units_column_conversion(snapshot_particles_path):
    collection = oc.open(snapshot_particles_path)
    converted = collection.with_units(sfr=u.Msun / u.Gyr, bhr=u.Msun / u.Gyr)
    sfr = converted["gas_particles"].select("sfr").get_data()
    bhr = converted["agn_particles"].select("bhr").get_data()
    assert sfr.unit == u.Msun / u.Gyr
    assert bhr.unit == u.Msun / u.Gyr
    assert np.allclose(
        sfr.value, _read(snapshot_particles_path, "gas_particles", "sfr") * 1e9
    )
    assert converted["dm_particles"].select("mass").get_data().unit == u.Msun


def test_with_units_unknown_column(snapshot_particles_path):
    collection = oc.open(snapshot_particles_path)
    with pytest.raises(ValueError, match="fof_halo_mass"):
        collection.with_units(fof_halo_mass=u.Msun)


@pytest.mark.parametrize("convention", ["scalefree", "comoving"])
def test_bound(snapshot_particles_path, snapshot_particle_writer, counts, convention):
    collection = oc.open(snapshot_particles_path).with_units(convention)
    scale = 1.0 if convention == "scalefree" else 1 / snapshot_particle_writer.h
    low = np.array([10.0, 20.0, 5.0])
    high = np.array([30.0, 45.0, 60.0])
    bounded = collection.bound(oc.make_box(tuple(low * scale), tuple(high * scale)))

    assert isinstance(bounded, oc.ParticleCollection)
    for name in counts:
        positions = np.stack(
            [_read(snapshot_particles_path, name, axis) for axis in "xyz"]
        )
        inside = np.all(
            (positions > low[:, None]) & (positions < high[:, None]), axis=0
        )
        cell_size = snapshot_particle_writer.box_size / (
            2**snapshot_particle_writer.max_level
        )
        cell_low = np.floor(positions / cell_size) * cell_size
        intersects = np.all(
            (cell_low < high[:, None]) & (cell_low + cell_size > low[:, None]),
            axis=0,
        )
        ids = _read(snapshot_particles_path, name, "id")
        expected = np.sort(ids[intersects])
        found = np.sort(bounded[name].select("id").get_data("numpy"))
        assert 0 < inside.sum() < len(found) < counts[name]
        assert np.all(np.isin(ids[inside], found))
        assert np.array_equal(found, expected)
        assert isinstance(bounded[name].index, tuple)

        exact = collection[name].bound(
            oc.make_box(tuple(low * scale), tuple(high * scale))
        )
        assert np.array_equal(
            np.sort(exact.select("id").get_data("numpy")), np.sort(ids[inside])
        )

    # bound returns a new collection and leaves the original unchanged
    for name, dataset in collection.items():
        assert len(dataset) == counts[name]


def test_bound_does_not_check_individual_particles(
    snapshot_particles_path, monkeypatch
):
    from opencosmo.spatial import check

    def unexpected_check(*args, **kwargs):
        pytest.fail("snapped bounds must not check individual particle coordinates")

    monkeypatch.setattr(check, "check_containment", unexpected_check)
    collection = oc.open(snapshot_particles_path).with_units("scalefree")
    bounded = collection.bound(oc.make_box((10, 20, 5), (30, 45, 60)))
    assert all(isinstance(dataset.index, tuple) for dataset in bounded.values())


def test_bound_again_within_region(snapshot_particles_path, snapshot_particle_writer):
    collection = oc.open(snapshot_particles_path).with_units("scalefree")
    outer = collection.bound(oc.make_box((10, 20, 5), (30, 45, 60)))
    inner = outer.bound(oc.make_box((12, 22, 8), (18, 28, 18)))
    cell_size = snapshot_particle_writer.box_size / (
        2**snapshot_particle_writer.max_level
    )
    low = np.array([12, 22, 8])
    high = np.array([18, 28, 18])

    for name in collection:
        positions = np.stack(
            [_read(snapshot_particles_path, name, axis) for axis in "xyz"]
        )
        cell_low = np.floor(positions / cell_size) * cell_size
        intersects = np.all(
            (cell_low < high[:, None]) & (cell_low + cell_size > low[:, None]), axis=0
        )
        expected = np.sort(_read(snapshot_particles_path, name, "id")[intersects])
        found = np.sort(inner[name].select("id").get_data("numpy"))
        assert np.array_equal(found, expected)
        assert isinstance(inner[name].index, tuple)
        assert np.all(np.isin(found, outer[name].select("id").get_data("numpy")))


def test_bound_disjoint_region(snapshot_particles_path):
    collection = oc.open(snapshot_particles_path).with_units("scalefree")
    bounded = collection.bound(oc.make_box((70, 70, 70), (80, 80, 80)))
    assert all(len(dataset) == 0 for dataset in bounded.values())


def test_empty_particle_type(tmp_path, snapshot_particle_writer, counts):
    path = snapshot_particle_writer.write(
        tmp_path / "early.hdf5", counts | {"agn_particles": 0}
    )
    collection = oc.open(path)
    assert len(collection["agn_particles"]) == 0
    bounded = collection.bound(oc.make_box((0, 0, 0), (20, 20, 20)))
    assert len(bounded["agn_particles"]) == 0
    assert len(bounded["dm_particles"]) > 0


def test_halo_particles_alone_still_raise(tmp_path, snapshot_particle_writer):
    path = snapshot_particle_writer.write(
        tmp_path / "haloparticles.hdf5", data_type="halo_particles"
    )
    with pytest.raises(ValueError, match="Cannot open particle data on its own"):
        oc.open(path)


def test_two_snapshot_particle_files_raise(tmp_path, snapshot_particle_writer):
    first = snapshot_particle_writer.write(tmp_path / "a.hdf5")
    second = snapshot_particle_writer.write(tmp_path / "b.hdf5")
    with pytest.raises(ValueError, match="cannot be combined"):
        oc.open(first, second)


def test_select_matches_columns_to_types(snapshot_particles_path, counts):
    collection = oc.open(snapshot_particles_path)
    selected = collection.select("x", "sfr", px=oc.col("mass") * oc.col("vx"))
    assert set(selected["gas_particles"].columns) == {"x", "sfr", "px"}
    for name in ("dm_particles", "star_particles", "agn_particles"):
        assert set(selected[name].columns) == {"x", "px"}
    data = selected.with_units("scalefree")["dm_particles"].get_data("numpy")
    mass = _read(snapshot_particles_path, "dm_particles", "mass")
    vx = _read(snapshot_particles_path, "dm_particles", "vx")
    assert np.allclose(data["px"], mass * vx)
    # select returns a new collection and leaves the original unchanged
    for name, dataset in collection.items():
        assert set(dataset.columns) == _fields(snapshot_particles_path, name)
        assert len(dataset) == counts[name]


def test_select_explicit(snapshot_particles_path):
    collection = oc.open(snapshot_particles_path)
    selected = collection.select(
        gas_particles=["rho"],
        dm_particles={
            "columns": ["x"],
            "derived_columns": {"px": oc.col("mass") * oc.col("vx")},
        },
    )
    assert set(selected["gas_particles"].columns) == {"rho"}
    assert set(selected["dm_particles"].columns) == {"x", "px"}
    assert set(selected["agn_particles"].columns) == set(
        collection["agn_particles"].columns
    )


def test_select_errors(snapshot_particles_path):
    collection = oc.open(snapshot_particles_path)
    with pytest.raises(ValueError, match="fof_halo_mass"):
        collection.select("fof_halo_mass")
    with pytest.raises(ValueError, match="halo_particles"):
        collection.select(gas_particles=["rho"], halo_particles=["x"])


def test_drop(snapshot_particles_path):
    collection = oc.open(snapshot_particles_path)
    dropped = collection.drop("sfr", "phi_grav", dm_particles="vx")
    gas_columns = set(collection["gas_particles"].columns)
    dm_columns = set(collection["dm_particles"].columns)
    assert set(dropped["gas_particles"].columns) == gas_columns - {"sfr", "phi_grav"}
    assert set(dropped["dm_particles"].columns) == dm_columns - {"phi_grav", "vx"}
    with pytest.raises(ValueError, match="halo_particles"):
        collection.drop(halo_particles="x")


def test_with_datasets(snapshot_particles_path):
    collection = oc.open(snapshot_particles_path)
    subset = collection.with_datasets(["gas_particles", "star_particles"])
    assert sorted(subset.keys()) == ["gas_particles", "star_particles"]
    assert sorted(collection.with_datasets("dm_particles").keys()) == ["dm_particles"]
    assert len(collection) == 4
    with pytest.raises(ValueError, match="halo_particles"):
        collection.with_datasets(["gas_particles", "halo_particles"])
    with pytest.raises(ValueError, match="at least one"):
        collection.with_datasets([])


def test_box_search(snapshot_particles_path, counts):
    collection = oc.open(snapshot_particles_path).with_units("scalefree")
    p1, p2 = (10.0, 20.0, 5.0), (30.0, 45.0, 60.0)
    searched = collection.box_search(p1, p2)
    bounded = collection.bound(oc.make_box(p1, p2))
    for name in counts:
        assert np.array_equal(
            searched[name].select("id").get_data("numpy"),
            bounded[name].select("id").get_data("numpy"),
        )


def test_evaluate_on_dataset(snapshot_particles_path):
    collection = oc.open(snapshot_particles_path).with_units("scalefree")

    def kinetic(mass, vx, vy, vz):
        return 0.5 * mass * (vx**2 + vy**2 + vz**2)

    evaluated = collection.evaluate_on_dataset(kinetic, "gas_particles")
    assert isinstance(evaluated, oc.ParticleCollection)
    assert "kinetic" in evaluated["gas_particles"].columns
    assert "kinetic" not in evaluated["dm_particles"].columns
    assert "kinetic" not in collection["gas_particles"].columns

    gas = collection["gas_particles"].select("mass", "vx", "vy", "vz")
    data = gas.get_data("numpy")
    expected = kinetic(data["mass"], data["vx"], data["vy"], data["vz"])
    inserted = evaluated["gas_particles"].select("kinetic").get_data("numpy")
    assert np.allclose(inserted, expected)

    batched = collection.evaluate_on_dataset(
        kinetic, "gas_particles", insert=False, format="numpy", batch_size=1000
    )
    assert np.allclose(batched["kinetic"], expected)

    with pytest.raises(ValueError, match="halo_particles"):
        collection.evaluate_on_dataset(kinetic, "halo_particles")


def test_get_data(snapshot_particles_path, counts):
    collection = oc.open(snapshot_particles_path).select("id")
    data = collection.get_data("numpy")
    assert sorted(data) == sorted(counts)
    for name in counts:
        assert np.array_equal(data[name], _read(snapshot_particles_path, name, "id"))


def test_write_round_trip(snapshot_particles_path, tmp_path, counts):
    region = oc.make_box((10.0, 20.0, 5.0), (30.0, 45.0, 60.0))
    collection = (
        oc.open(snapshot_particles_path)
        .with_units("scalefree")
        .bound(region)
        .select("id", "x", "y", "z", px=oc.col("mass") * oc.col("vx"))
    )
    path = tmp_path / "written.hdf5"
    oc.write(path, collection)

    reopened = oc.open(path).with_units("scalefree")
    assert isinstance(reopened, oc.ParticleCollection)
    assert sorted(reopened.keys()) == sorted(counts)
    for name in counts:
        assert set(reopened[name].columns) == {"id", "x", "y", "z", "px"}
        written = reopened[name].select("id", "px").get_data("numpy")
        expected = collection[name].select("id", "px").get_data("numpy")
        order, expected_order = np.argsort(written["id"]), np.argsort(expected["id"])
        assert np.array_equal(written["id"][order], expected["id"][expected_order])
        assert np.allclose(written["px"][order], expected["px"][expected_order])
    # the written spatial index still answers spatial queries
    inner = oc.make_box((15.0, 25.0, 10.0), (25.0, 40.0, 50.0))
    for name in counts:
        assert np.array_equal(
            np.sort(reopened.bound(inner)[name].select("id").get_data("numpy")),
            np.sort(collection.bound(inner)[name].select("id").get_data("numpy")),
        )
