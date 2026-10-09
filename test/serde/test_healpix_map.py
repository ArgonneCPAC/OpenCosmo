import numpy as np
import pytest
from opencosmo.serde import (
    ArithmeticExpression,
    ColumnReference,
    ComparisonMask,
    ConeRegionMessage,
    DropMessage,
    FilterMessage,
    HealpixBoundMessage,
    HealpixMapMessage,
    NumberLiteral,
    SelectMessage,
    SerdeError,
    SortByMessage,
    TakeMessage,
    TakeRangeMessage,
    TakeRowsMessage,
    WithNewColumnsMessage,
    apply_message,
)
from pydantic import TypeAdapter, ValidationError

import opencosmo as oc


@pytest.fixture
def healpix_map(test_data):
    return oc.open(test_data.healpix_map)


def test_healpix_map_message_union():
    adapter = TypeAdapter(HealpixMapMessage)

    assert isinstance(
        adapter.validate_python({"kind": "select", "columns": ["tsz"]}),
        SelectMessage,
    )
    assert isinstance(
        adapter.validate_python(
            {
                "kind": "healpix_bound",
                "region": {
                    "kind": "cone",
                    "center": [45.0, -45.0],
                    "radius": 2.0,
                },
            }
        ),
        HealpixBoundMessage,
    )


def test_healpix_bound_message_rejects_unsupported_region():
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        HealpixBoundMessage.model_validate(
            {
                "kind": "healpix_bound",
                "region": {
                    "kind": "box",
                    "p1": [0, 0, 0],
                    "p2": [1, 1, 1],
                },
            }
        )


def test_healpix_map_select_drop_and_derive(healpix_map):
    selected = apply_message(healpix_map, SelectMessage(columns=("tsz",)))
    dropped = apply_message(healpix_map, DropMessage(columns=("ksz",)))
    derived = apply_message(
        healpix_map,
        WithNewColumnsMessage(
            columns={
                "tsz2": ArithmeticExpression(
                    operator="power",
                    left=ColumnReference(name="tsz"),
                    right=NumberLiteral(value=2),
                )
            }
        ),
    )

    assert selected.columns == ["tsz"]
    assert dropped.columns == ["tsz"]
    assert "tsz2" in derived.columns
    assert np.array_equal(selected.pixels, healpix_map.pixels)


def test_healpix_map_filter_updates_coverage(healpix_map):
    result = apply_message(
        healpix_map,
        FilterMessage(
            masks=(
                ComparisonMask(
                    operator="greater_than",
                    left=ColumnReference(name="tsz"),
                    right=NumberLiteral(value=0.0),
                ),
            )
        ),
    )

    assert len(result) <= len(healpix_map)
    assert set(result.pixels).issubset(set(healpix_map.pixels))
    assert result.full_sky is False


def test_healpix_map_take_messages_update_pixels(healpix_map):
    start = apply_message(healpix_map, TakeMessage(n=10, at="start"))
    ranged = apply_message(healpix_map, TakeRangeMessage(start=3, end=8))
    rows = apply_message(healpix_map, TakeRowsMessage(rows=(1, 3, 5)))

    assert np.array_equal(start.pixels, healpix_map.pixels[:10])
    assert np.array_equal(ranged.pixels, healpix_map.pixels[3:8])
    assert np.array_equal(rows.pixels, healpix_map.pixels[[1, 3, 5]])


def test_healpix_map_rejects_incompatible_shared_options(healpix_map):
    cases = (
        (TakeMessage(n=2, mode="global"), "does not support global mode"),
        (
            TakeRangeMessage(start=0, end=2, mode="global"),
            "does not support global mode",
        ),
        (TakeRowsMessage(rows=()), "requires at least one row"),
        (SortByMessage(column=None), "requires a column"),
        (SelectMessage(columns=("tsz",), mode="local"), "does not support local mode"),
        (
            WithNewColumnsMessage(
                columns={"tsz2": ColumnReference(name="tsz")},
                allow_overwrite=True,
            ),
            "does not support overwrite",
        ),
    )
    for message, expected in cases:
        result = apply_message(healpix_map, message)
        assert isinstance(result, SerdeError)
        assert expected in result.message


def test_healpix_map_bound_message(healpix_map):
    result = apply_message(
        healpix_map,
        HealpixBoundMessage(region=ConeRegionMessage(center=(45.0, -45.0), radius=2.0)),
    )
    expected = healpix_map.bound(oc.make_cone((45.0, -45.0), 2.0))

    assert np.array_equal(result.pixels, expected.pixels)
    assert set(result.pixels).issubset(set(healpix_map.pixels))


def test_healpix_map_sort_then_take(healpix_map):
    result = apply_message(
        apply_message(healpix_map, SortByMessage(column="tsz", invert=True)),
        TakeMessage(n=5, at="start"),
    )

    assert len(result) == 5
    assert len(result.pixels) == 5
