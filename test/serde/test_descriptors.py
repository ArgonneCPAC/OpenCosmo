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
def test_messages_keyed_by_method(name):
    descriptor = DESCRIPTORS[name]
    assert descriptor.type_name == name
    for method, models in descriptor.allowed_messages.items():
        assert models
        assert len(descriptor.signatures[method]) == len(models)
        for model in models:
            assert model.method == method
            kind = model.model_fields["kind"].default
            assert descriptor.messages_by_kind[kind] is model


def test_methods_use_plain_method_names():
    structure = DESCRIPTORS["StructureCollection"].allowed_messages
    assert "select" in structure
    assert not any(method.startswith("structure_") for method in structure)
    assert "bound" in DESCRIPTORS["HealpixMap"].allowed_messages


def test_simulation_methods_have_several_candidates():
    messages = DESCRIPTORS["SimulationCollection"].allowed_messages
    kinds = {m.model_fields["kind"].default for m in messages["select"]}
    assert kinds == {"select", "structure_select"}
    assert len(messages["with_new_columns"]) == 3


def test_lightcone_extends_dataset():
    dataset = DESCRIPTORS["Dataset"].allowed_messages
    lightcone = DESCRIPTORS["Lightcone"].allowed_messages
    assert dataset.items() <= lightcone.items()
    assert {"with_redshift_range", "pixel_search"} <= lightcone.keys()


@pytest.mark.parametrize("name", list(DESCRIPTORS))
def test_type_name_matches_summary_kind(name):
    assert DESCRIPTORS[name].summary_type.model_fields["kind"].default == name
