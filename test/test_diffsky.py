import matplotlib.pyplot as plt
import numpy as np
import pytest

import opencosmo as oc
from opencosmo.analysis import get_pop_mah


@pytest.fixture
def core_path_487(diffsky_path):
    return diffsky_path / "lj_487.hdf5"


@pytest.fixture
def core_path_475(diffsky_path):
    return diffsky_path / "lj_475.hdf5"


def test_comoving_to_physical(core_path_487):
    cores = oc.open(core_path_487, synth_cores=True).select(["redshift_true", "x"])
    data_physical = cores.with_units("physical").select(["redshift_true", "x"]).data
    data_comoving = cores.select(["redshift_true", "x"]).data
    a = 1 / (data_physical["redshift_true"] + 1)
    assert np.all(np.isclose(data_physical["x"], data_comoving["x"] * a))


def test_comoving_to_scalefree(core_path_487):
    with pytest.raises(oc.transformations.units.UnitError):
        _ = oc.open(core_path_487, synth_cores=True).with_units("scalefree")


def test_comoving_to_unitless(core_path_487):
    ds = oc.open(core_path_487, synth_cores=True)
    data = ds.data
    data_unitless = ds.with_units("unitless").data
    for col in data.columns:
        assert np.all(data[col].value == data_unitless[col].value)


def test_mah_pop(core_path_487):
    zs = np.linspace(0, 1, 10)
    ds = oc.open(core_path_487, synth_cores=False)
    c = get_pop_mah(ds, zs)
    plt.plot(c[0], c[1])
    plt.savefig("test.png")
    assert False


def test_open_multiple(core_path_487, core_path_475):
    ds = oc.open(core_path_487, core_path_475, synth_cores=True)
    assert len(ds.keys()) == 4
