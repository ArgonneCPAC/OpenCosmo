from pathlib import Path

import astropy.cosmology as ac
import h5py
import pytest

from opencosmo import read_cosmology


@pytest.fixture
def cosmology_resource_path(data_path):
    p = data_path / "galaxyproperties.hdf5"
    return p


@pytest.fixture
def open_lcdm(cosmology_resource_path, tmp_path):
    update = {"Ode0": 0.8}
    return update_cosmology_parameter(
        cosmology_resource_path, update, tmp_path, "open_lcdm"
    )


@pytest.fixture
def flat_wcdm(cosmology_resource_path, tmp_path):
    update = {"w_0": -0.9}
    return update_cosmology_parameter(
        cosmology_resource_path, update, tmp_path, "flat_wcdm"
    )


@pytest.fixture
def closed_wcdm(cosmology_resource_path, tmp_path):
    update = {"w_0": -0.9, "Ode0": 0.6}
    return update_cosmology_parameter(
        cosmology_resource_path, update, tmp_path, "closed_wcdm"
    )


@pytest.fixture
def flat_wowa(cosmology_resource_path, tmp_path):
    update = {"w_0": -0.9, "w_a": 0.1}
    return update_cosmology_parameter(
        cosmology_resource_path, update, tmp_path, "flat_wowa"
    )


@pytest.fixture
def open_wowa(cosmology_resource_path, tmp_path):
    update = {"w_0": -0.9, "w_a": 0.1, "Ode0": 0.6}
    return update_cosmology_parameter(
        cosmology_resource_path, update, tmp_path, "open_wowa"
    )


def update_cosmology_parameter(
    base_cosmology_path: Path, parameters: dict[str, float], temp_path: Path, name: str
):
    # make a copy of the original data
    path = temp_path / f"{name}.hdf5"
    with h5py.File(base_cosmology_path, "r") as f:
        with h5py.File(path, "w") as file:
            f.copy(f["header"], file, "header")
            # update the attributes
            for key, value in parameters.items():
                file["header"]["simulation"]["cosmology"].attrs[key] = value
    return path


def test_flat_lcdm(cosmology_resource_path):
    cosmo = read_cosmology(cosmology_resource_path)
    assert isinstance(cosmo, ac.FlatLambdaCDM)


def test_open_lcdm(open_lcdm):
    cosmo = read_cosmology(open_lcdm)
    assert isinstance(cosmo, ac.LambdaCDM)


def test_flat_wcdm(flat_wcdm):
    cosmo = read_cosmology(flat_wcdm)
    assert isinstance(cosmo, ac.FlatwCDM)


def test_closed_wcdm(closed_wcdm):
    cosmo = read_cosmology(closed_wcdm)
    assert isinstance(cosmo, ac.wCDM)


def test_flat_wowa(flat_wowa):
    cosmo = read_cosmology(flat_wowa)
    assert isinstance(cosmo, ac.Flatw0waCDM)


def test_open_wowa(open_wowa):
    cosmo = read_cosmology(open_wowa)
    assert isinstance(cosmo, ac.w0waCDM)
