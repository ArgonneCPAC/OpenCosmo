import pytest
from astropy.table import Table
import numpy as np

import opencosmo as oc
from opencosmo.analysis import create_yt_dataset


@pytest.fixture
def particle_data(data_path):
    """Fetch particle data for a single halo"""
    return oc.read(data_path / "haloparticles.hdf5").take(1, at="start")[0]


def test_create_dataset(particle_data):
    """Check that a yt dataset is created without X-ray fields."""
    ds = create_yt_dataset(particle_data)
    assert ds is not None
    assert hasattr(ds, "all_data")


def test_create_dataset_with_xray_fields(particle_data):
    """Check that X-ray fields are added and source model is returned."""
    ds, source_model = create_yt_dataset(
        particle_data,
        compute_xray_fields=True,
        return_source_model=True
    )

    assert ds is not None
    assert source_model is not None
    assert hasattr(source_model, "make_source_fields")


def test_xray_fields_present(particle_table):
    """Verify that key X-ray fields are registered in yt."""

    ds = create_yt_dataset(
        particle_data,
        compute_xray_fields=True,
        return_source_model=True,
        source_model_kwargs={
            "emin": 0.5,
            "emax": 2.0,
            "nbins": 50,
        }
    )

    required_fields = [
        ("gas", "xray_emissivity_0.5_2.0_keV"),
        ("gas", "xray_luminosity_0.5_2.0_keV"),
    ]

    for field in required_fields:
        assert field in ds.derived_field_list
