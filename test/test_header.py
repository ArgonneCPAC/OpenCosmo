from pathlib import Path

import astropy.units as u
import h5py
import pytest
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import units as cu

from opencosmo.header import read_header, write_header


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
    data = data.with_units("scalefree")
    assert data.simulation["box_size"] == 128.0 * u.Mpc / cu.littleh


def test_read_header(header_resource_path):
    header = read_header(header_resource_path)
    assert isinstance(header.cosmology, FlatLambdaCDM)


def test_write_header(header_resource_path, tmp_path):
    header = read_header(header_resource_path)
    write_header(tmp_path / "header.hdf5", header)
    header = read_header(header_resource_path)

    assert isinstance(header.cosmology, FlatLambdaCDM)


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
