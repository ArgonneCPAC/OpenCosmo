import numpy as np
import pytest

import opencosmo as oc


@pytest.fixture
def core_path(diffsky_path):
    return diffsky_path / "lj_487.hdf5"


def test_comoving_to_physical(core_path):
    cores = oc.open(core_path, synth_cores=True).select(["redshift_true", "x"])
    data_physical = cores.with_units("physical").select(["redshift_true", "x"]).data
    data_comoving = cores.select(["redshift_true", "x"]).data
    a = 1 / (data_physical["redshift_true"] + 1)
    assert np.all(np.isclose(data_physical["x"], data_comoving["x"] * a))


def test_comoving_to_scalefree(core_path):
    with pytest.raises(oc.transformations.units.UnitError):
        _ = oc.open(core_path, synth_cores=True).with_units("scalefree")


def test_comoving_to_unitless(core_path):
    ds = oc.open(core_path, synth_cores=True)
    data = ds.data
    data_unitless = ds.with_units("unitless").data
    for col in data.columns:
        assert np.all(data[col].value == data_unitless[col].value)
