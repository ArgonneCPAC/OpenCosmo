import pytest
from opencosmo.serde import (
    ArithmeticExpression,
    ClearMatchMessage,
    ColumnReference,
    ComparisonMask,
    DropMessage,
    FilterMessage,
    MatchMessage,
    NumberLiteral,
    SelectMessage,
    SerdeError,
    SimulationMessage,
    SimulationWithNewColumnsMessage,
    StructureFilterMessage,
    StructureWithNewColumnsMessage,
    TakeRangeMessage,
    WithUnitsMessage,
    apply_message,
)
from opencosmo.utils import normalize_kwarg_name
from pydantic import TypeAdapter, ValidationError

import opencosmo as oc

REFERENCE = normalize_kwarg_name("SCIDAC_128_GO")
SIMULATION_A = normalize_kwarg_name(
    "KAPPA_2.222_EGW_0.759_SEED_7.810e5_VKIN_5889_EPS_5.257"
)


@pytest.fixture
def dataset_collection(test_data):
    gravity = oc.open(test_data.snapshot.primary.halo_properties)
    hydro = oc.open(test_data.snapshot.primary.galaxy_properties)
    return oc.SimulationCollection({"gravity": gravity, "hydro": hydro})


@pytest.fixture
def structure_collection(test_data):
    first = oc.open(*test_data.snapshot.primary.halos)
    second = oc.open(*test_data.snapshot.primary.halos)
    return oc.SimulationCollection({"first": first, "second": second})


@pytest.fixture
def mapped_collection(test_data):
    return oc.open(
        test_data.snapshot.mapping_reference,
        test_data.snapshot.scidac(0).halo_properties,
        test_data.snapshot.halo_mapping,
    )


def test_simulation_message_union_includes_shared_and_control_messages():
    adapter = TypeAdapter(SimulationMessage)

    assert isinstance(
        adapter.validate_python({"kind": "select", "columns": ["mass"]}),
        SelectMessage,
    )
    assert isinstance(
        adapter.validate_python({"kind": "match", "dataset": "gravity"}),
        MatchMessage,
    )
    assert isinstance(
        adapter.validate_python({"kind": "clear_match"}), ClearMatchMessage
    )


@pytest.mark.parametrize("name", ["", ".gravity", "gravity.run"])
def test_simulation_names_reject_invalid_values(name):
    with pytest.raises(ValidationError):
        MatchMessage(dataset=name)


def test_simulation_dataset_select_and_drop_route_by_schema(dataset_collection):
    selected = apply_message(
        dataset_collection,
        SelectMessage(columns=("fof_halo_mass", "gal_mass_star")),
    )
    dropped = apply_message(
        dataset_collection,
        DropMessage(columns=("fof_halo_mass", "gal_mass_star")),
    )

    assert set(selected["gravity"].columns) == {"fof_halo_mass"}
    assert set(selected["hydro"].columns) == {"gal_mass_star"}
    assert "fof_halo_mass" not in dropped["gravity"].columns
    assert "gal_mass_star" not in dropped["hydro"].columns


def test_simulation_dataset_filter_and_units_delegate(dataset_collection):
    # Use a column available in only one child by targeting a homogeneous collection.
    gravity = dataset_collection["gravity"]
    collection = oc.SimulationCollection({"a": gravity, "b": gravity})
    filtered = apply_message(
        collection,
        FilterMessage(
            masks=(
                ComparisonMask(
                    operator="greater_than",
                    left=ColumnReference(name="fof_halo_mass"),
                    right=NumberLiteral(value=1e13),
                ),
            )
        ),
    )
    converted = apply_message(filtered, WithUnitsMessage(convention="physical"))

    assert all(len(child) <= len(gravity) for child in converted.values())
    assert all(
        child._state.convention.value == "physical" for child in converted.values()
    )


def test_simulation_structure_filter_delegates(structure_collection):
    result = apply_message(
        structure_collection,
        StructureFilterMessage(
            masks=(
                ComparisonMask(
                    operator="greater_than",
                    left=ColumnReference(name="sod_halo_mass"),
                    right=NumberLiteral(value=1e13),
                ),
            )
        ),
    )

    assert all(
        len(child) <= len(structure_collection[name]) for name, child in result.items()
    )


def test_simulation_with_new_columns_targets_children(dataset_collection):
    result = apply_message(
        dataset_collection,
        SimulationWithNewColumnsMessage(
            datasets=("gravity",),
            columns={
                "mass2": ArithmeticExpression(
                    operator="multiply",
                    left=ColumnReference(name="fof_halo_mass"),
                    right=NumberLiteral(value=2.0),
                )
            },
        ),
    )

    assert "mass2" in result["gravity"].columns
    assert "mass2" not in result["hydro"].columns


def test_simulation_structure_new_columns_delegate(structure_collection):
    result = apply_message(
        structure_collection,
        StructureWithNewColumnsMessage(
            dataset="halo_properties",
            columns={
                "mass2": ArithmeticExpression(
                    operator="multiply",
                    left=ColumnReference(name="fof_halo_mass"),
                    right=NumberLiteral(value=2.0),
                )
            },
        ),
    )

    assert all("mass2" in child["halo_properties"].columns for child in result.values())


def test_simulation_match_and_source_controlled_range(mapped_collection):
    matched = apply_message(mapped_collection, MatchMessage(dataset=REFERENCE))
    result = apply_message(matched, TakeRangeMessage(start=2, end=9))

    source_tags = result[REFERENCE].select("fof_halo_tag").get_data("numpy")
    target_tags = result[SIMULATION_A].select("fof_halo_tag").get_data("numpy")
    assert len(source_tags) == len(target_tags) == 7

    cleared = apply_message(result, ClearMatchMessage())
    assert set(cleared.keys()) == {REFERENCE, SIMULATION_A}


def test_simulation_match_rejects_unknown_source(mapped_collection):
    result = apply_message(mapped_collection, MatchMessage(dataset="unknown"))

    assert isinstance(result, SerdeError)
    assert "does not have a simulation named unknown" in result.message
