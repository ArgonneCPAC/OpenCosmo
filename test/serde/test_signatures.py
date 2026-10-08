"""Message models must mirror the signatures of the methods they represent."""

import inspect

import pytest
from opencosmo.collection.lightcone.healpix_map import HealpixMap
from opencosmo.collection.lightcone.lightcone import Lightcone
from opencosmo.collection.simulation.simulation import SimulationCollection
from opencosmo.collection.structure.structure import StructureCollection
from opencosmo.dataset.dataset import Dataset
from opencosmo.serde import DESCRIPTORS, ParameterRole

TARGETS = {
    "Dataset": Dataset,
    "Lightcone": Lightcone,
    "HealpixMap": HealpixMap,
    "StructureCollection": StructureCollection,
    "SimulationCollection": SimulationCollection,
}

# Messages that cannot be compared parameter by parameter.
SKIPPED = {
    ("StructureCollection", "structure_select"): "dataset-keyed kwargs",
    ("StructureCollection", "structure_drop"): "dataset-keyed kwargs",
    ("SimulationCollection", "filter"): "forwards *args/**kwargs",
    ("SimulationCollection", "structure_filter"): "forwards *args/**kwargs",
    ("SimulationCollection", "select"): "forwards *args/**kwargs",
    ("SimulationCollection", "structure_select"): "forwards *args/**kwargs",
    ("SimulationCollection", "drop"): "forwards *args/**kwargs",
    ("SimulationCollection", "structure_drop"): "forwards *args/**kwargs",
    ("SimulationCollection", "structure_with_units"): "not delegable",
    ("SimulationCollection", "with_new_columns"): "keyword-only in this method",
    ("SimulationCollection", "structure_with_new_columns"): "keyword-only here",
    ("HealpixMap", "drop"): "takes a single positional argument",
}

# Message fields the method does not accept.
UNSUPPORTED_FIELDS = {
    ("HealpixMap", "filter"): {"mode"},
    ("HealpixMap", "select"): {"mode"},
    ("HealpixMap", "take"): {"mode"},
    ("HealpixMap", "take_range"): {"mode"},
    ("HealpixMap", "with_new_columns"): {"allow_overwrite", "mode"},
    # The optional leading dataset name travels in the method's *args.
    ("SimulationCollection", "simulation_with_new_columns"): {"dataset"},
}

# Method parameters that exist only to forward arguments to children.
IGNORED_METHOD_PARAMS = {
    ("SimulationCollection", "simulation_with_new_columns"): {"args"},
    ("Lightcone", "filter"): {"kwargs"},
    ("HealpixMap", "filter"): {"kwargs"},
}

ROLES = {
    inspect.Parameter.POSITIONAL_OR_KEYWORD: ParameterRole.POSITIONAL,
    inspect.Parameter.VAR_POSITIONAL: ParameterRole.VAR_ARGS,
    inspect.Parameter.KEYWORD_ONLY: ParameterRole.KEYWORD_ONLY,
    inspect.Parameter.VAR_KEYWORD: ParameterRole.VAR_KWARGS,
}

CASES = [
    (target, method, index)
    for target, descriptor in DESCRIPTORS.items()
    for method, models in descriptor.allowed_messages.items()
    for index, model in enumerate(models)
    if (target, model.model_fields["kind"].default) not in SKIPPED
]


@pytest.mark.parametrize(("target", "method_name", "index"), CASES)
def test_message_matches_method_signature(target, method_name, index):
    kind = (
        DESCRIPTORS[target]
        .allowed_messages[method_name][index]
        .model_fields["kind"]
        .default
    )
    method = getattr(TARGETS[target], method_name)
    forwarded = IGNORED_METHOD_PARAMS.get((target, kind), set())
    expected = [
        (p.name, ROLES[p.kind])
        for p in inspect.signature(method).parameters.values()
        if p.name != "self" and p.name not in forwarded
    ]
    ignored = UNSUPPORTED_FIELDS.get((target, kind), set())
    actual = [
        (p.name, p.role)
        for p in DESCRIPTORS[target].signatures[method_name][index]
        if p.name not in ignored
    ]

    assert len(actual) == len(expected)
    for (name, role), (method_name, method_role) in zip(actual, expected):
        assert role == method_role, (name, method_name)
        if role in (ParameterRole.POSITIONAL, ParameterRole.KEYWORD_ONLY):
            assert name == method_name


def test_skips_and_ignores_reference_real_messages():
    for target, kind in [*SKIPPED, *UNSUPPORTED_FIELDS, *IGNORED_METHOD_PARAMS]:
        assert kind in DESCRIPTORS[target].messages_by_kind
