import pytest

import opencosmo as oc
from opencosmo.analysis import create_yt_dataset, visualize_halo


@pytest.fixture
def data(snapshot_path):
    """Fetch particle data for a couple of halos"""
    haloproperties = snapshot_path / "haloproperties.hdf5"
    haloparticles = snapshot_path / "haloparticles.hdf5"
    d = (
        oc.open_linked_files([haloproperties, haloparticles])
        .filter(oc.col("sod_halo_mass") > 5e13)
        .take(3, at="start")
    )

    assert len(list(d.objects())) > 0

    return d


def test_create_dataset(data):
    """Check that a yt dataset is created without X-ray fields."""

    for halo in data.halos():
        ds = create_yt_dataset(halo)

        assert ds is not None
        assert hasattr(ds, "all_data")


def test_create_dataset_with_xray_fields(data):
    """Check that X-ray fields are added and source model is returned."""

    for halo in data.halos():
        ds, source_model = create_yt_dataset(
            halo, compute_xray_fields=True, return_source_model=True
        )

        assert ds is not None
        assert source_model is not None
        assert hasattr(source_model, "make_source_fields")


def test_xray_fields_present(data):
    """Verify that key X-ray fields are registered in yt."""

    for halo in data.halos():
        ds = create_yt_dataset(
            halo,
            compute_xray_fields=True,
            source_model_kwargs={
                "emin": 0.5,
                "emax": 2.0,
                "nbins": 50,
            },
        )

        required_fields = [
            ("gas", "xray_emissivity_0.5_2.0_keV"),
            ("gas", "xray_luminosity_0.5_2.0_keV"),
        ]

        for field in required_fields:
            assert field in ds.derived_field_list


def test_multipanel_visualization(data):
    """Check that yt visualization tool works"""

    for halo in data.halos():
        halo_id = halo['halo_properties']['fof_halo_tag']
        visualize_halo(halo_id, data).savefig(f"{halo_id}.png")
