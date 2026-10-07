import numpy as np
import pytest
from opencosmo.serde import (
    ArithmeticExpression,
    ColumnReference,
    ComparisonMask,
    NumberLiteral,
    PixelSearchMessage,
    SerdeError,
    StructureDropMessage,
    StructureDropTarget,
    StructureFilterMessage,
    StructureMessage,
    StructureSelectionTarget,
    StructureSelectMessage,
    StructureUnitTarget,
    StructureWithNewColumnsMessage,
    StructureWithUnitsMessage,
    TakeMessage,
    WithDatasetsMessage,
    WithRedshiftRangeMessage,
    apply_message,
)
from pydantic import TypeAdapter, ValidationError

import opencosmo as oc


@pytest.fixture
def halo_collection(test_data):
    return oc.open(*test_data.snapshot.primary.halos)


@pytest.fixture
def lightcone_halo_collection(test_data):
    return oc.open(
        *test_data.lightcone.step(600).halos,
        *test_data.lightcone.step(601).halos,
    )


def test_structure_message_union_includes_shared_and_specific_messages():
    adapter = TypeAdapter(StructureMessage)

    assert isinstance(
        adapter.validate_python({"kind": "take", "n": 2}),
        TakeMessage,
    )
    assert isinstance(
        adapter.validate_python(
            {
                "kind": "structure_filter",
                "masks": [],
                "on_galaxies": True,
            }
        ),
        StructureFilterMessage,
    )
    assert isinstance(
        adapter.validate_python(
            {"kind": "with_datasets", "datasets": ["halo_properties"]}
        ),
        WithDatasetsMessage,
    )


@pytest.mark.parametrize(
    "path", ["", ".galaxies", "galaxies.", "galaxies..star_particles"]
)
def test_structure_messages_reject_invalid_dataset_paths(path):
    with pytest.raises(ValidationError):
        WithDatasetsMessage(datasets=(path,))


def test_structure_select_rejects_mixed_routing():
    with pytest.raises(ValidationError, match="cannot be combined with targets"):
        StructureSelectMessage(
            columns=("fof_halo_mass",),
            targets={
                "halo_properties": StructureSelectionTarget(columns=("fof_halo_tag",))
            },
        )


def test_structure_filter_and_take_match_direct_operations(halo_collection):
    message_result = apply_message(
        apply_message(
            halo_collection,
            StructureFilterMessage(
                masks=(
                    ComparisonMask(
                        operator="greater_than",
                        left=ColumnReference(name="sod_halo_mass"),
                        right=NumberLiteral(value=1e13),
                    ),
                )
            ),
        ),
        TakeMessage(n=10, at="start"),
    )
    expected = halo_collection.filter(oc.col("sod_halo_mass") > 1e13).take(
        10, at="start"
    )

    actual_tags = message_result["halo_properties"].select("fof_halo_tag").get_data()
    expected_tags = expected["halo_properties"].select("fof_halo_tag").get_data()
    assert np.array_equal(actual_tags, expected_tags)


def test_structure_select_routes_columns_automatically(halo_collection):
    result = apply_message(
        halo_collection,
        StructureSelectMessage(columns=("sod_halo_mass", "fof_halo_bin_tag")),
    )

    assert set(result["halo_properties"].columns) == {"sod_halo_mass"}
    assert set(result["halo_profiles"].columns) == {"fof_halo_bin_tag"}


def test_structure_select_and_drop_target_datasets(halo_collection):
    selected = apply_message(
        halo_collection,
        StructureSelectMessage(
            targets={
                "halo_properties": StructureSelectionTarget(
                    columns=("fof_halo_tag", "fof_halo_mass")
                ),
                "halo_profiles": StructureSelectionTarget(
                    columns=("fof_halo_bin_tag",)
                ),
            }
        ),
    )
    dropped = apply_message(
        selected,
        StructureDropMessage(
            targets={"halo_properties": StructureDropTarget(columns=("fof_halo_mass",))}
        ),
    )

    assert set(selected["halo_properties"].columns) == {
        "fof_halo_tag",
        "fof_halo_mass",
    }
    assert set(selected["halo_profiles"].columns) == {"fof_halo_bin_tag"}
    assert set(dropped["halo_properties"].columns) == {"fof_halo_tag"}


def test_structure_with_new_columns_targets_dataset(halo_collection):
    result = apply_message(
        halo_collection,
        StructureWithNewColumnsMessage(
            dataset="halo_properties",
            columns={
                "mass2": ArithmeticExpression(
                    operator="multiply",
                    left=ColumnReference(name="fof_halo_mass"),
                    right=NumberLiteral(value=2.0),
                )
            },
            descriptions={"mass2": "Twice the FOF mass"},
        ),
    )

    source = result["halo_properties"]
    assert "mass2" in source.columns
    assert source.descriptions["mass2"] == "Twice the FOF mass"


def test_structure_with_units_targets_dataset(halo_collection):
    result = apply_message(
        halo_collection,
        StructureWithUnitsMessage(
            datasets={
                "halo_properties": StructureUnitTarget(columns={"fof_halo_mass": "kg"})
            }
        ),
    )

    assert result["halo_properties"].units["fof_halo_mass"].is_equivalent("kg")


def test_structure_with_datasets_limits_visible_members(halo_collection):
    result = apply_message(
        halo_collection,
        WithDatasetsMessage(datasets=("halo_properties", "halo_profiles")),
    )

    assert set(result.keys()) == {"halo_properties", "halo_profiles"}


def test_lightcone_structure_specific_messages(lightcone_halo_collection):
    restricted = apply_message(
        lightcone_halo_collection,
        WithRedshiftRangeMessage(z_low=0.038, z_high=0.039),
    )
    redshifts = restricted["halo_properties"].select("redshift").get_data()
    assert np.all((redshifts > 0.038) & (redshifts < 0.039))

    pixels = lightcone_halo_collection.get_pixels(64)
    searched = apply_message(
        lightcone_halo_collection,
        PixelSearchMessage(pixels=frozenset(pixels[:5].tolist()), nside=64),
    )
    assert len(searched) <= len(lightcone_halo_collection)


def test_snapshot_structure_rejects_lightcone_specific_message(halo_collection):
    result = apply_message(
        halo_collection,
        WithRedshiftRangeMessage(z_low=0.03, z_high=0.04),
    )

    assert isinstance(result, SerdeError)
    assert result.exception_type == "AttributeError"
