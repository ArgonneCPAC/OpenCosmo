"""Tests for the per-type message descriptors."""

import pytest
from opencosmo.serde import DESCRIPTORS, DataClassDescriptor


def test_descriptor_types():
    assert set(DESCRIPTORS) == {
        "Dataset",
        "Lightcone",
        "HealpixMap",
        "StructureCollection",
        "SimulationCollection",
    }
    assert all(isinstance(d, DataClassDescriptor) for d in DESCRIPTORS.values())


@pytest.mark.parametrize("name", list(DESCRIPTORS))
def test_messages_keyed_by_kind(name):
    descriptor = DESCRIPTORS[name]
    assert descriptor.type_name == name
    for kind, model in descriptor.allowed_messages.items():
        assert model.model_fields["kind"].default == kind


def test_lightcone_extends_dataset():
    dataset = DESCRIPTORS["Dataset"].allowed_messages
    lightcone = DESCRIPTORS["Lightcone"].allowed_messages
    assert dataset.items() <= lightcone.items()
    assert {"with_redshift_range", "pixel_search"} <= lightcone.keys()
