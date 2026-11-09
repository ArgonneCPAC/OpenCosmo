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
        expected = np.sort(_read(snapshot_particles_path, name, "id")[inside])
        found = np.sort(bounded[name].select("id").get_data("numpy"))
        assert 0 < len(found) < counts[name]
        assert np.array_equal(found, expected)

    # bound returns a new collection and leaves the original unchanged
    for name, dataset in collection.items():
        assert len(dataset) == counts[name]


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
