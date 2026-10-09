import healpy as hp
import numpy as np
import pytest
from opencosmo.serde import (
    ArithmeticExpression,
    BoundMessage,
    BoxRegionMessage,
    ColumnReference,
    ComparisonMask,
    FilterMessage,
    LightconeMessage,
    NumberLiteral,
    PixelSearchMessage,
    SelectMessage,
    SerdeError,
    SortByMessage,
    TakeMessage,
    WithNewColumnsMessage,
    WithRedshiftRangeMessage,
    apply_message,
)
from pydantic import TypeAdapter, ValidationError

import opencosmo as oc


@pytest.fixture
def lightcone(test_data):
    return oc.open(
        test_data.lightcone.step(600).halo_properties,
        test_data.lightcone.step(601).halo_properties,
    )


def test_lightcone_message_union_includes_shared_and_specific_messages():
    adapter = TypeAdapter(LightconeMessage)

    assert isinstance(adapter.validate_python({"kind": "take", "n": 2}), TakeMessage)
    assert isinstance(
        adapter.validate_python(
            {"kind": "with_redshift_range", "z_low": 0.04, "z_high": 0.041}
        ),
        WithRedshiftRangeMessage,
    )
    assert isinstance(
        adapter.validate_python({"kind": "pixel_search", "pixels": [1], "nside": 2}),
        PixelSearchMessage,
    )


def test_redshift_message_rejects_empty_interval():
    with pytest.raises(ValidationError, match="must differ"):
        WithRedshiftRangeMessage(z_low=0.04, z_high=0.04)


def test_pixel_search_message_validates_and_sorts_pixels():
    message = PixelSearchMessage(pixels=frozenset({3, 1, 2}), nside=2)

    assert message.model_dump(mode="json")["pixels"] == [1, 2, 3]
    with pytest.raises(ValidationError, match="at least 1 item"):
        PixelSearchMessage(pixels=frozenset(), nside=2)
    with pytest.raises(ValidationError, match="positive power of two"):
        PixelSearchMessage(pixels=frozenset({1}), nside=3)
    with pytest.raises(ValidationError, match="pixels must be less than"):
        PixelSearchMessage(pixels=frozenset({48}), nside=2)


def test_messages_transform_an_actual_lightcone(lightcone):
    mass = "fof_halo_mass"
    tag = "fof_halo_tag"
    filtered = apply_message(
        lightcone,
        FilterMessage(
            masks=(
                ComparisonMask(
                    operator="greater_than",
                    left=ColumnReference(name=mass),
                    right=NumberLiteral(value=1e13),
                ),
            )
        ),
    )
    derived = apply_message(
        filtered,
        WithNewColumnsMessage(
            columns={
                "mass2": ArithmeticExpression(
                    operator="multiply",
                    left=ColumnReference(name=mass),
                    right=NumberLiteral(value=2.0),
                )
            }
        ),
    )
    selected = apply_message(
        derived,
        SelectMessage(columns=(tag, "mass2")),
    )
    result = apply_message(
        apply_message(selected, SortByMessage(column=tag, invert=True)),
        TakeMessage(n=2, at="start"),
    ).get_data()

    expected = (
        lightcone.filter(oc.col(mass) > 1e13)
        .with_new_columns(mass2=oc.col(mass) * 2.0)
        .select(tag, "mass2")
        .sort_by(tag, invert=True)
        .take(2, at="start")
        .get_data()
    )

    assert np.array_equal(result[tag], expected[tag])
    assert np.array_equal(result["mass2"], expected["mass2"])


def test_apply_redshift_range_message(lightcone):
    message = WithRedshiftRangeMessage(z_low=0.04, z_high=0.0405)

    result = apply_message(lightcone, message)
    expected = lightcone.with_redshift_range(0.04, 0.0405)

    assert result.z_range == expected.z_range == (0.04, 0.0405)
    assert np.array_equal(
        result.select("fof_halo_tag").get_data(),
        expected.select("fof_halo_tag").get_data(),
    )


def test_apply_pixel_search_message(lightcone):
    pixels = lightcone.get_pixels(64)
    selected_pixels = frozenset(pixels[: min(5, len(pixels))].tolist())
    message = PixelSearchMessage(pixels=selected_pixels, nside=64)

    result = apply_message(lightcone, message)
    coordinates = result.select("theta", "phi").get_data("numpy")
    found_pixels = np.unique(
        hp.ang2pix(
            64,
            coordinates["theta"],
            coordinates["phi"],
            nest=True,
        )
    )

    assert set(found_pixels).issubset(selected_pixels)


def test_lightcone_rejects_three_dimensional_bound(lightcone):
    result = apply_message(
        lightcone,
        BoundMessage(region=BoxRegionMessage(p1=(0, 0, 0), p2=(1, 1, 1))),
    )

    assert isinstance(result, SerdeError)
    assert "Three-dimensional box" in result.message


def test_dataset_rejects_lightcone_specific_message(test_data):
    dataset = oc.open(test_data.snapshot.primary.halo_properties)

    result = apply_message(
        dataset,
        WithRedshiftRangeMessage(z_low=0.04, z_high=0.0405),  # type: ignore[arg-type]
    )

    assert isinstance(result, SerdeError)
    assert "only supported by Lightcone" in result.message
