import pytest


@pytest.fixture
def halo_properties_path(test_data):
    return test_data.snapshot.primary.halo_properties


@pytest.fixture
def haloproperties_600_path(test_data):
    return test_data.lightcone.step(600).halo_properties


@pytest.fixture
def haloproperties_601_path(test_data):
    return test_data.lightcone.step(601).halo_properties
