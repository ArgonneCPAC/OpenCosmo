import shutil

import astropy.units as u
import h5py
import numpy as np
import opencosmo as oc
import pytest
from astropy.cosmology import units as cu

HYDRO_COMPONENTS = {"total", "cdm", "bar", "gas", "star", "sfgas", "wind", "agn"}


@pytest.fixture
def hydro(test_data):
    return oc.open(test_data.power_spectrum.hydro)


@pytest.fixture
def hydro_copy(test_data, tmp_path):
    path = tmp_path / "hydro.hdf5"
    shutil.copy(test_data.power_spectrum.hydro, path)
    return path


def test_open(hydro):
    assert isinstance(hydro, oc.PowerSpectrum)
    assert set(hydro.components) == HYDRO_COMPONENTS
    n_steps = len(hydro.steps)
    n_k = len(hydro.n_modes)
    assert np.all(np.diff(hydro.steps) > 0)
    assert hydro.redshifts.shape == (n_steps,)
    assert hydro.k.shape == (n_k,)
    assert np.all(np.diff(hydro.k) > 0)
    for component in hydro.components:
        assert hydro.get_data(component)[component].shape == (n_steps, n_k)


def test_open_gravity_only(test_data):
    ps = oc.open(test_data.power_spectrum.gravity_only)
    assert ps.components == ("total",)
    assert ps.get_data("total")["total"].shape == (len(ps.steps), len(ps.k))


def test_header(hydro):
    assert hydro.header.file.data_type == "power_spectrum"
    assert hydro.header.file.step is None
    assert hydro.power_spectrum["grid"] == hydro.simulation["pm_grid"]
    assert hydro.power_spectrum["binning"] == "linear"
    assert hydro.cosmology.h > 0


def test_comoving_units(hydro):
    assert hydro.k.unit == u.Mpc**-1
    assert hydro.get_data("cdm")["cdm"].unit == u.Mpc**3


def test_scalefree_units(hydro):
    scalefree = hydro.with_units("scalefree")
    h = hydro.cosmology.h
    assert scalefree.k.unit == cu.littleh / u.Mpc
    assert scalefree.get_data("total")["total"].unit == (u.Mpc / cu.littleh) ** 3
    assert np.allclose(scalefree.k.value * h, hydro.k.value)
    assert np.allclose(
        scalefree.get_data("total")["total"].value / h**3,
        hydro.get_data("total")["total"].value,
    )


def test_physical_units_are_refused(hydro):
    with pytest.raises(ValueError, match="no physical units"):
        hydro.with_units("physical")


def test_unitless(hydro):
    unitless = hydro.with_units("unitless")
    assert not isinstance(unitless.k, u.Quantity)
    assert not isinstance(unitless.get_data("total")["total"], u.Quantity)


def test_with_units_returns_new_object(hydro):
    scalefree = hydro.with_units("scalefree")
    assert scalefree is not hydro
    assert hydro.k.unit == u.Mpc**-1


def test_get_step(hydro):
    full = hydro.get_data("cdm")["cdm"]
    for i, step in enumerate(hydro.steps):
        assert np.array_equal(hydro.get_data("cdm", step=step), full[i])


def test_get_redshift(hydro):
    i = len(hydro.steps) // 2
    redshift = hydro.redshifts[i] + 0.005
    single = hydro.get_data("total", redshift=redshift)
    assert single.shape == (len(hydro.k),)
    all_steps = hydro.get_data("total")
    assert np.array_equal(single, all_steps["total"][i])


def test_get_multiple_redshifts(hydro):
    redshifts = list(hydro.redshifts[::3] + 0.005)
    data = hydro.get_data("total", redshift=redshifts)
    assert data["total"].shape == (len(redshifts), len(hydro.k))
    assert "redshift" in data.colnames
    assert np.allclose(data["redshift"], hydro.redshifts[::3])
    for row, i in enumerate(range(0, len(hydro.steps), 3)):
        assert np.array_equal(
            data["total"][row], hydro.get_data("total", step=hydro.steps[i])
        )


def test_redshift_column_in_all_steps_output(hydro):
    data = hydro.get_data()
    assert np.allclose(data["redshift"], hydro.redshifts)
    single = hydro.get_data("cdm", step=hydro.steps[0])
    assert single.shape == (len(hydro.k),)


def test_redshift_outside_tolerance(hydro):
    redshifts = hydro.redshifts
    steps = hydro.steps
    # Steps are in increasing order, so redshifts decrease; pick a gap wider than 0.02
    i = int(np.argmax(-np.diff(redshifts)))
    redshift = (redshifts[i] + redshifts[i + 1]) / 2
    message = (
        rf"within 0.01 of redshift {redshift}\. The closest are "
        rf"step {steps[i + 1]} \(z = {redshifts[i + 1]:.4f}\) and "
        rf"step {steps[i]} \(z = {redshifts[i]:.4f}\)"
    )
    with pytest.raises(ValueError, match=message):
        hydro.get_data(redshift=redshift)


def test_redshift_beyond_last_step(hydro):
    with pytest.raises(
        ValueError, match=r"The closest are step 624 \(z = 0\.0000\)\.$"
    ):
        hydro.get_data(redshift=-0.5)


def test_get_all_components(hydro):
    from astropy.table import QTable

    data = hydro.get_data()
    assert isinstance(data, QTable)
    assert set(data.colnames) == set(hydro.components) | {"redshift"}
    for name in hydro.components:
        assert data[name].shape == (len(hydro.steps), len(hydro.k))
        assert np.all(data[name].unit == u.Mpc**3)


def test_get_component_list(hydro):
    data = hydro.get_data(["cdm", "gas"], step=hydro.steps[0])
    assert set(data.colnames) == {"cdm", "gas"}
    assert data["cdm"].shape == (len(hydro.k),)
    assert np.array_equal(
        data["cdm"].value, hydro.get_data("cdm", step=hydro.steps[0]).value
    )


def test_get_data_numpy_format(hydro):
    data = hydro.get_data(format="numpy")
    assert isinstance(data, dict)
    assert not isinstance(data["cdm"], u.Quantity)
    single = hydro.get_data("cdm", step=hydro.steps[0], format="numpy")
    assert isinstance(single, np.ndarray)


def test_get_data_unknown_format(hydro):
    with pytest.raises(ValueError, match="Unknown data output format"):
        hydro.get_data(format="parquet")


def test_get_data_empty_component_list(hydro):
    with pytest.raises(ValueError, match="at least one component"):
        hydro.get_data([])


def test_unknown_component(hydro):
    with pytest.raises(ValueError, match="no component 'dm'"):
        hydro.get_data("dm")


def test_unknown_step(hydro):
    with pytest.raises(ValueError, match="Step 1 is not in this power spectrum"):
        hydro.get_data(step=1)


def test_step_and_redshift(hydro):
    with pytest.raises(ValueError, match="either a step or a redshift"):
        hydro.get_data(step=hydro.steps[0], redshift=0)


def test_nonfinite_redshift(hydro):
    with pytest.raises(ValueError, match="must be finite"):
        hydro.get_data(redshift=np.nan)


def test_axes_are_read_only(hydro):
    for name in ("steps", "redshifts", "n_modes"):
        axis = getattr(hydro, name)
        with pytest.raises(ValueError, match="read-only"):
            axis[0] = 0
        with pytest.raises(ValueError, match="WRITEABLE"):
            axis.flags.writeable = True


def test_returned_values_are_read_only(hydro):
    unitless = hydro.with_units("unitless")
    for value in (
        unitless.k,
        unitless.get_data("total")["total"],
        unitless.get_data("cdm", step=unitless.steps[0]),
    ):
        assert not value.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            value[:] = 0


def test_missing_spectrum_is_nan(hydro_copy):
    with h5py.File(hydro_copy, "r+") as f:
        f["power_spectrum/pk_agn"][0] = np.nan
    ps = oc.open(hydro_copy)
    assert np.all(np.isnan(ps.get_data("agn", step=ps.steps[0])))
    assert not np.any(np.isnan(ps.get_data("agn")["agn"][1:]))


def test_missing_axis(hydro_copy):
    with h5py.File(hydro_copy, "r+") as f:
        del f["power_spectrum/n_modes"]
    with pytest.raises(ValueError, match="missing the datasets n_modes"):
        oc.open(hydro_copy)


def test_wrong_shape(hydro_copy):
    with h5py.File(hydro_copy, "r+") as f:
        pk = f["power_spectrum/pk_cdm"][:]
        del f["power_spectrum/pk_cdm"]
        f["power_spectrum/pk_cdm"] = pk[:, :-1]
    with pytest.raises(ValueError, match="pk_cdm must have the shape"):
        oc.open(hydro_copy)


def test_open_with_other_simulation(test_data):
    with pytest.raises(TypeError, match="Simulation collection only accepts"):
        oc.open(test_data.power_spectrum.hydro, test_data.power_spectrum.gravity_only)
