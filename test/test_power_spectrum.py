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
    data = hydro.get_data()
    for component in hydro.components:
        assert data[component].shape == (n_steps, n_k)


def test_open_gravity_only(test_data):
    ps = oc.open(test_data.power_spectrum.gravity_only)
    assert ps.components == ("total",)
    assert ps.get_data()["total"].shape == (len(ps.steps), len(ps.k))


def test_header(hydro):
    assert hydro.header.file.data_type == "power_spectrum"
    assert hydro.header.file.step is None
    assert hydro.power_spectrum["grid"] == hydro.simulation["pm_grid"]
    assert hydro.power_spectrum["binning"] == "linear"
    assert hydro.cosmology.h > 0


def test_parameters(hydro):
    parameters = hydro.parameters
    assert isinstance(parameters, dict)
    assert parameters == hydro.header.parameters["power_spectrum"]
    assert parameters["grid"] == hydro.simulation["pm_grid"]
    parameters["grid"] = -1
    assert hydro.parameters["grid"] == hydro.simulation["pm_grid"]


def test_parameter_units(hydro):
    wavenumbers = ("k_fundamental", "k_nyquist", "k_max", "bin_k_min", "bin_width")
    h = hydro.cosmology.h
    comoving = hydro.parameters
    scalefree = hydro.with_units("scalefree").parameters
    unitless = hydro.with_units("unitless").parameters
    for name in wavenumbers:
        assert comoving[name].unit == hydro.k.unit
        assert scalefree[name].unit == cu.littleh / u.Mpc
        assert np.isclose(scalefree[name].value * h, comoving[name].value)
        assert not isinstance(unitless[name], u.Quantity)
    assert np.all(hydro.k <= comoving["k_max"])


def test_comoving_units(hydro):
    assert hydro.k.unit == u.Mpc**-1
    assert hydro.get_data()["cdm"].unit == u.Mpc**3


def test_scalefree_units(hydro):
    scalefree = hydro.with_units("scalefree")
    h = hydro.cosmology.h
    assert scalefree.k.unit == cu.littleh / u.Mpc
    assert scalefree.get_data()["total"].unit == (u.Mpc / cu.littleh) ** 3
    assert np.allclose(scalefree.k.value * h, hydro.k.value)
    assert np.allclose(
        scalefree.get_data()["total"].value / h**3,
        hydro.get_data()["total"].value,
    )


def test_physical_units_are_refused(hydro):
    with pytest.raises(ValueError, match="no physical units"):
        hydro.with_units("physical")


def test_unitless(hydro):
    unitless = hydro.with_units("unitless")
    assert not isinstance(unitless.k, u.Quantity)
    assert not isinstance(unitless.get_data()["total"], u.Quantity)


def test_with_units_returns_new_object(hydro):
    scalefree = hydro.with_units("scalefree")
    assert scalefree is not hydro
    assert hydro.k.unit == u.Mpc**-1


def test_select(hydro):
    selected = hydro.select("cdm", "gas")
    assert selected is not hydro
    assert selected.components == ("cdm", "gas")
    assert set(hydro.components) == HYDRO_COMPONENTS
    full = hydro.get_data()
    data = selected.get_data()
    assert set(data.colnames) == {"cdm", "gas", "redshift"}
    for name in ("cdm", "gas"):
        assert np.array_equal(data[name], full[name])


def test_select_all(hydro):
    assert hydro.select("all").components == hydro.components
    assert hydro.select(list(hydro.components)).components == hydro.components


def test_drop(hydro):
    dropped = hydro.drop("agn", ["wind", "star"])
    assert dropped is not hydro
    assert "agn" not in dropped.components
    assert "wind" not in dropped.components
    assert "star" not in dropped.components
    assert set(dropped.components) == HYDRO_COMPONENTS - {"agn", "wind", "star"}
    full = hydro.get_data()
    data = dropped.get_data()
    for name in dropped.components:
        assert np.array_equal(data[name], full[name])


def test_drop_everything(hydro):
    with pytest.raises(ValueError, match="Cannot drop every component"):
        hydro.drop("all")


def test_with_steps(hydro):
    steps = [int(hydro.steps[0]), int(hydro.steps[-1])]
    selected = hydro.with_steps(steps)
    assert selected is not hydro
    assert list(selected.steps) == steps
    assert np.allclose(selected.redshifts, hydro.redshifts[[0, -1]])
    full = hydro.select("total").get_data()["total"]
    data = selected.select("total").get_data()["total"]
    for row, i in enumerate((0, -1)):
        assert np.array_equal(data[row], full[i])


def test_with_redshifts(hydro):
    i = len(hydro.steps) // 2
    redshift = hydro.redshifts[i] + 0.005
    single = hydro.with_redshifts(redshift)
    assert single.steps.shape == (1,)
    # A single component at a single step is returned on its own.
    assert np.array_equal(
        single.select("total").get_data(),
        hydro.select("total").get_data()["total"][i : i + 1],
    )


def test_with_multiple_redshifts(hydro):
    redshifts = list(hydro.redshifts[::3] + 0.005)
    selected = hydro.with_redshifts(redshifts)
    data = selected.select("total").get_data()
    assert data["total"].shape == (len(redshifts), len(hydro.k))
    assert np.allclose(data["redshift"], hydro.redshifts[::3])
    full = hydro.select("total").get_data()
    for row, i in enumerate(range(0, len(hydro.steps), 3)):
        assert np.array_equal(data["total"][row], full["total"][i])


def test_redshift_column_in_all_steps_output(hydro):
    data = hydro.get_data()
    assert np.allclose(data["redshift"], hydro.redshifts)
    # A single step has no "redshift" column, and a single component at a
    # single step is returned on its own.
    single = hydro.select("cdm").with_steps(hydro.steps[0]).get_data()
    assert single.shape == (1, len(hydro.k))
    # Several components at a single step still have no "redshift" column.
    pair = hydro.select("cdm", "gas").with_steps(hydro.steps[0]).get_data()
    assert set(pair.colnames) == {"cdm", "gas"}


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
        hydro.with_redshifts(redshift)


def test_redshift_tolerance(hydro):
    redshifts = hydro.redshifts
    i = int(np.argmax(-np.diff(redshifts)))
    redshift = (redshifts[i] + redshifts[i + 1]) / 2
    gap = redshifts[i] - redshifts[i + 1]
    selected = hydro.with_redshifts(redshift, tolerance=gap)
    assert selected.steps[0] in hydro.steps[i : i + 2]
    with pytest.raises(ValueError, match=r"within 0\.001 of redshift"):
        hydro.with_redshifts(redshift, tolerance=0.001)


@pytest.mark.parametrize("tolerance", [-0.1, np.nan, np.inf])
def test_invalid_redshift_tolerance(hydro, tolerance):
    with pytest.raises(ValueError, match="tolerance must be finite and non-negative"):
        hydro.with_redshifts(0, tolerance=tolerance)


def test_redshift_beyond_last_step(hydro):
    with pytest.raises(
        ValueError, match=r"The closest are step 624 \(z = 0\.0000\)\.$"
    ):
        hydro.with_redshifts(-0.5)


def test_get_all_components(hydro):
    from astropy.table import QTable

    data = hydro.get_data()
    assert isinstance(data, QTable)
    assert set(data.colnames) == set(hydro.components) | {"redshift"}
    for name in hydro.components:
        assert data[name].shape == (len(hydro.steps), len(hydro.k))
        assert np.all(data[name].unit == u.Mpc**3)


def test_get_data_numpy_format(hydro):
    data = hydro.get_data(format="numpy")
    assert isinstance(data, dict)
    assert not isinstance(data["cdm"], u.Quantity)
    single = hydro.select("cdm").with_steps(hydro.steps[0]).get_data(format="numpy")
    assert isinstance(single, np.ndarray)


def test_get_data_unknown_format(hydro):
    with pytest.raises(ValueError, match="Unknown data output format"):
        hydro.get_data(format="parquet")


def test_select_empty(hydro):
    with pytest.raises(ValueError, match="at least one component"):
        hydro.select([])


def test_unknown_component(hydro):
    with pytest.raises(ValueError, match="no component 'dm'"):
        hydro.select("dm")


def test_unknown_component_drop(hydro):
    with pytest.raises(ValueError, match="no component 'dm'"):
        hydro.drop("dm")


@pytest.mark.parametrize("args", [(), ([],)])
def test_empty_step_selection(hydro, args):
    with pytest.raises(ValueError, match="at least one step"):
        hydro.with_steps(*args)
    with pytest.raises(ValueError, match="at least one step"):
        hydro.with_redshifts(*args)


def test_unknown_step(hydro):
    with pytest.raises(ValueError, match="Step 1 is not in this power spectrum"):
        hydro.with_steps(1)


def test_nonfinite_redshift(hydro):
    with pytest.raises(ValueError, match="must be finite"):
        hydro.with_redshifts(np.nan)


def test_axes_are_read_only(hydro):
    for name in ("steps", "redshifts", "n_modes"):
        axis = getattr(hydro, name)
        with pytest.raises(ValueError, match="read-only"):
            axis[0] = 0
        with pytest.raises(ValueError, match="WRITEABLE"):
            axis.flags.writeable = True


def test_selected_axes_are_read_only(hydro):
    for ps in (
        hydro.with_steps(list(hydro.steps[:2])),
        hydro.with_redshifts(list(hydro.redshifts[:2])),
    ):
        for name in ("steps", "redshifts"):
            axis = getattr(ps, name)
            with pytest.raises(ValueError, match="read-only"):
                axis[0] = 0
            with pytest.raises(ValueError, match="WRITEABLE"):
                axis.flags.writeable = True


def test_returned_values_are_read_only(hydro):
    unitless = hydro.with_units("unitless")
    for value in (
        unitless.k,
        unitless.get_data()["total"],
        unitless.select("cdm").with_steps(unitless.steps[0]).get_data(),
    ):
        assert not value.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            value[:] = 0


def test_missing_spectrum_is_nan(hydro_copy):
    with h5py.File(hydro_copy, "r+") as f:
        f["power_spectrum/pk_agn"][0] = np.nan
    ps = oc.open(hydro_copy)
    single = ps.select("agn").with_steps(ps.steps[0]).get_data()
    assert np.all(np.isnan(single))
    assert not np.any(np.isnan(ps.get_data()["agn"][1:]))


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
