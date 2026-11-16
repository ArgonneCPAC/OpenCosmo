from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.cosmology as ac
import h5py
import pytest
from opencosmo.cosmology import read_cosmology

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def cosmology_resource_path(test_data):
    return test_data.snapshot.primary.galaxy_properties


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
    with h5py.File(base_cosmology_path, "r") as f, h5py.File(path, "w") as file:
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


BASE_PARAMETERS = {
    "h": 0.7,
    "omega_m": 0.3,
    "omega_b": 0.05,
    "omega_l": 0.7,
    "n_eff_massless": 3.04,
    "sigma_8": 0.8,
    "w_0": -1.0,
    "w_a": 0.0,
}


def test_cosmology_without_cmb_temperature():
    from opencosmo.dtypes.cosmology import CosmologyParameters, make_cosmology

    parameters = CosmologyParameters.model_validate(BASE_PARAMETERS)
    cosmology = make_cosmology(parameters)

    assert isinstance(cosmology, ac.FlatLambdaCDM)
    assert cosmology.Tcmb0.value == 0.0
    assert "Tcmb0" not in parameters.model_dump(by_alias=True, exclude_none=True)


def test_cosmology_cmb_temperature_is_passed_to_astropy():
    from opencosmo.dtypes.cosmology import CosmologyParameters, make_cosmology

    parameters = CosmologyParameters.model_validate(BASE_PARAMETERS | {"Tcmb0": 2.7255})
    cosmology = make_cosmology(parameters)
    expected = ac.FlatLambdaCDM(H0=70.0, Om0=0.3, Tcmb0=2.7255, Neff=3.04)

    assert isinstance(cosmology, ac.FlatLambdaCDM)
    assert cosmology.Tcmb0.value == 2.7255
    assert cosmology.Ogamma0 + cosmology.Onu0 == expected.Ogamma0 + expected.Onu0
    assert cosmology.Ode0 == pytest.approx(1 - 0.3 - expected.Ogamma0 - expected.Onu0)
    assert parameters.model_dump(by_alias=True)["Tcmb0"] == 2.7255


def test_nonflat_cosmology_cmb_temperature_contributes_curvature():
    from opencosmo.dtypes.cosmology import CosmologyParameters, make_cosmology

    parameters = CosmologyParameters.model_validate(
        BASE_PARAMETERS | {"omega_l": 0.6, "Tcmb0": 2.7255}
    )
    cosmology = make_cosmology(parameters)

    assert isinstance(cosmology, ac.LambdaCDM)
    radiation = cosmology.Ogamma0 + cosmology.Onu0
    assert radiation > 0
    assert cosmology.Ok0 == pytest.approx(1 - 0.3 - 0.6 - radiation)
