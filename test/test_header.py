from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import h5py
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import units as cu
from opencosmo.header import read_header, write_header
from opencosmo.spatial.region import HealpixRegion

import opencosmo as oc

if TYPE_CHECKING:
    from pathlib import Path


def update_simulation_parameter(
    base_cosmology_path: Path, parameters: dict[str, float], tmp_path: Path, name: str
):
    # make a copy of the original data
    path = tmp_path / f"{name}.hdf5"
    with h5py.File(base_cosmology_path, "r") as f:
        with h5py.File(path, "w") as file:
            f.copy(f["header"], file, "header")
            # update the attributes
            for key, value in parameters.items():
                file["header"]["simulation"]["parameters"].attrs[key] = value
    return path


@pytest.fixture
def header_resource_path(snapshot_path):
    p = snapshot_path / "galaxyproperties.hdf5"
    return p


@pytest.fixture
def malformed_header_path(header_resource_path, tmp_path):
    update = {"n_dm": "foo"}
    return update_simulation_parameter(
        header_resource_path, update, tmp_path, "malformed_header"
    )


def test_header_units(header_resource_path):
    data = read_header(header_resource_path)
    assert data.simulation["box_size"] == (128.0 / data.cosmology.h) * u.Mpc
    assert data.simulation["agn_seed_mass"].unit == u.Msun
    data = data.with_units("scalefree")
    assert data.simulation["box_size"] == 128.0 * u.Mpc / cu.littleh
    assert data.simulation["agn_seed_mass"].unit == u.Msun / cu.littleh


def test_read_header(header_resource_path):
    header = read_header(header_resource_path)
    assert isinstance(header.cosmology, FlatLambdaCDM)


def test_write_header(header_resource_path, tmp_path):
    path = tmp_path / "header.hdf5"
    header = read_header(header_resource_path)
    write_header(path, header)
    header = read_header(path)

    assert isinstance(header.cosmology, FlatLambdaCDM)


def test_write_header_with_large_array(header_resource_path, tmp_path):
    region_model = HealpixRegion(np.arange(0, 2**14), 2**15)
    path = tmp_path / "header.hdf5"

    header = read_header(header_resource_path)
    header = header.with_region(region_model)
    write_header(path, header)
    header = read_header(path)


def test_malformed_header(malformed_header_path):
    with pytest.raises(ValueError):
        read_header(malformed_header_path)


def test_simulation_step_to_redshift(header_resource_path):
    header = read_header(header_resource_path)
    simulation_pars = header.simulation
    step_zs = simulation_pars["step_zs"]
    assert step_zs[205] == 2.004
    assert step_zs[-1] == 0.0
    assert all(step_zs[i] > step_zs[i + 1] for i in range(len(step_zs) - 1))


# ---------------------------------------------------------------------------
# HaccSimulationInfo / simulation_info tests
# ---------------------------------------------------------------------------


def test_simulation_info_galaxyproperties(header_resource_path):
    """simulation["name"] returns a non-empty str on galaxyproperties.hdf5."""
    header = read_header(header_resource_path)
    name = header.simulation["name"]
    assert isinstance(name, str)
    assert name != ""


def test_simulation_info_haloproperties(snapshot_path):
    """simulation["name"] returns the known value for haloproperties.hdf5."""
    header = read_header(snapshot_path / "haloproperties.hdf5")
    assert (
        header.simulation["name"]
        == "KAPPA_2_EGW_0.568_SEED_1.048e6_VKIN_7984_EPS_10.130"
    )


def test_simulation_info_haloproperties_go(snapshot_path):
    """simulation["name"] returns the known value for haloproperties_go.hdf5."""
    header = read_header(snapshot_path / "haloproperties_go.hdf5")
    assert header.simulation["name"] == "SCIDAC_128_GO"


def test_simulation_info_roundtrip(snapshot_path, tmp_path):
    """simulation["name"] is preserved after write/reopen (regression for silent drop)."""
    src = snapshot_path / "haloproperties.hdf5"
    dest = tmp_path / "haloproperties_roundtrip.hdf5"
    dataset = oc.open(src)
    oc.write(dest, dataset)
    reopened = oc.open(dest)
    assert reopened.header.simulation["name"] == dataset.header.simulation["name"]


def test_simulation_info_does_not_disturb_subgroups(header_resource_path):
    """Registering 'simulation' as an optional HDF5 path does not shadow the required
    entries read from simulation/parameters, simulation/cosmotools, and
    simulation/cosmology.  read_header_attributes (dtypes/parameters.py ~L26-28) skips
    subgroups when iterating header_group.items(), which is why both can coexist.
    """
    header = read_header(header_resource_path)
    # simulation_info (the new optional block) resolves.
    name = header.simulation["name"]
    assert isinstance(name, str) and name != ""
    # simulation/parameters still resolves via the required 'simulation' ACCESS_PATH.
    assert header.simulation["box_size"] is not None
    # simulation/cosmology still produces a valid astropy cosmology.
    assert isinstance(header.cosmology, FlatLambdaCDM)
