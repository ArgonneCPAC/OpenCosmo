import pytest


@pytest.fixture
def halo_properties_path(snapshot_path):
    return snapshot_path / "haloproperties.hdf5"


@pytest.fixture
def haloproperties_600_path(lightcone_path):
    return lightcone_path / "step_600" / "haloproperties.hdf5"


@pytest.fixture
def haloproperties_601_path(lightcone_path):
    return lightcone_path / "step_601" / "haloproperties.hdf5"
