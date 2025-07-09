import pytest

import opencosmo as oc


@pytest.fixture
def core_path(diffsky_path):
    return diffsky_path / "lj_487.hdf5"


def test_cores(core_path):
    cores = oc.open(core_path).with_units("physical")
    print(cores.select("redshift").data)
    assert False
