import operator

import astropy.units as u
import numpy as np
import pytest
from opencosmo.column.column import (
    Column,
    ColumnMask,
    CompoundColumnMask,
)
from opencosmo.serde import (
    ArithmeticExpression,
    BoundMessage,
    BoxRegionMessage,
    ColumnReference,
    ComparisonMask,
    ConeRegionMessage,
    DatasetMessage,
    DropMessage,
    FilterMessage,
    HealpixRegionMessage,
    MembershipMask,
    NumberLiteral,
    QuantityLiteral,
    ReductionExpression,
    SelectMessage,
    SerdeError,
    SkyboxRegionMessage,
    SortByMessage,
    TakeMessage,
    TakeRangeMessage,
    TakeRowsMessage,
    WithNewColumnsMessage,
    WithUnitsMessage,
    apply_message,
)
from opencosmo.serde.rehydrate.expression import expression_to_live, mask_to_live
from pydantic import TypeAdapter, ValidationError

import opencosmo as oc
from opencosmo.dataset import Dataset


def test_dataset_messages_validate_by_discriminator():
    adapter = TypeAdapter(DatasetMessage)

    filter_message = adapter.validate_python(
        {
            "kind": "filter",
            "masks": [
                {
                    "kind": "comparison",
                    "operator": "greater_than",
                    "left": {"kind": "column", "name": "mass"},
                    "right": {"kind": "number", "value": 1},
                }
            ],
        }
    )
    select_message = adapter.validate_python(
        {
            "kind": "select",
            "columns": ["tag"],
            "derived_columns": {
                "scaled_mass": {
                    "kind": "arithmetic",
                    "operator": "multiply",
                    "left": {"kind": "column", "name": "mass"},
                    "right": {"kind": "number", "value": 2},
                }
            },
            "mode": "local",
        }
    )

    assert isinstance(filter_message, FilterMessage)
    assert isinstance(select_message, SelectMessage)
    assert select_message.mode.value == "local"


@pytest.mark.parametrize(
    ("message", "error"),
    [
        (
            {"kind": "filter", "mode": "rank", "masks": []},
            "Input should be 'local' or 'global'",
        ),
        (
            {"kind": "select", "columns": [""], "derived_columns": {}},
            "Column names and patterns must not be empty",
        ),
        (
            {
                "kind": "select",
                "columns": [],
                "derived_columns": {"": {"kind": "column", "name": "mass"}},
            },
            "Derived column names must not be empty",
        ),
    ],
)
def test_dataset_messages_reject_invalid_inputs(message, error):
    with pytest.raises(ValidationError, match=error):
        TypeAdapter(DatasetMessage).validate_python(message)


def test_expression_to_live_builds_lazy_expression():
    expression = ArithmeticExpression(
        operator="divide",
        left=ArithmeticExpression(
            operator="subtract",
            left=ColumnReference(name="mass"),
            right=ReductionExpression(
                operator="mean", operand=ColumnReference(name="mass")
            ),
        ),
        right=ReductionExpression(operator="std", operand=ColumnReference(name="mass")),
    )

    live = expression_to_live(expression)

    assert isinstance(live, Column)
    assert live.requires_names == {"mass"}
    values = np.array([1.0, 2.0, 3.0])
    assert np.allclose(
        live.evaluate({"mass": values}), (values - values.mean()) / values.std()
    )


def test_expression_to_live_preserves_quantity():
    live = expression_to_live(QuantityLiteral(value=2.0, unit="Mpc"))

    assert live == 2.0 * u.Mpc


def test_mask_to_live_builds_comparison_and_membership_masks():
    comparison = mask_to_live(
        ComparisonMask(
            operator="greater_than",
            left=ColumnReference(name="mass"),
            right=NumberLiteral(value=2),
        )
    )
    membership = mask_to_live(
        MembershipMask(
            operand=ColumnReference(name="tag"),
            values=(NumberLiteral(value=1), NumberLiteral(value=3)),
        )
    )

    assert isinstance(comparison, ColumnMask)
    assert comparison.requires_names == {"mass"}
    assert comparison.apply({"mass": np.array([1, 2, 3])}).tolist() == [
        False,
        False,
        True,
    ]
    assert membership.apply({"tag": np.array([1, 2, 3])}).tolist() == [
        True,
        False,
        True,
    ]


def test_mask_to_live_builds_compound_mask():
    message = TypeAdapter(DatasetMessage).validate_python(
        {
            "kind": "filter",
            "masks": [
                {
                    "kind": "compound",
                    "operator": "and",
                    "left": {
                        "kind": "comparison",
                        "operator": "greater_than",
                        "left": {"kind": "column", "name": "mass"},
                        "right": {"kind": "number", "value": 1},
                    },
                    "right": {
                        "kind": "comparison",
                        "operator": "less_than",
                        "left": {"kind": "column", "name": "mass"},
                        "right": {"kind": "number", "value": 3},
                    },
                }
            ],
        }
    )

    live = mask_to_live(message.masks[0])

    assert isinstance(live, CompoundColumnMask)
    assert live.apply({"mass": np.array([1, 2, 3])}).tolist() == [False, True, False]


def test_expression_to_live_rejects_invalid_scalar_contexts():
    with pytest.raises(ValueError, match="mean requires a vector expression"):
        expression_to_live(
            ReductionExpression(operator="mean", operand=NumberLiteral(value=1))
        )

    with pytest.raises(ValueError, match="requires a vector expression"):
        mask_to_live(
            ComparisonMask(
                operator="equal",
                left=NumberLiteral(value=1),
                right=NumberLiteral(value=1),
            )
        )


class RecordingDataset(Dataset):
    def __init__(self):
        self.call = None

    def filter(self, *masks, mode):
        self.call = ("filter", masks, mode)
        return self

    def select(self, *columns, mode, **derived_columns):
        self.call = ("select", columns, mode, derived_columns)
        return self

    def drop(self, *columns):
        self.call = ("drop", columns)
        return self

    def sort_by(self, column, invert=False):
        self.call = ("sort_by", column, invert)
        return self

    def take(self, n, at="random", mode="local"):
        self.call = ("take", n, at, mode)
        return self

    def take_range(self, start, end, mode="local"):
        self.call = ("take_range", start, end, mode)
        return self

    def take_rows(self, rows):
        self.call = ("take_rows", rows)
        return self

    def bound(self, region, select_by=None):
        self.call = ("bound", region, select_by)
        return self

    def with_new_columns(
        self, descriptions, allow_overwrite=False, mode="global", **columns
    ):
        self.call = (
            "with_new_columns",
            descriptions,
            allow_overwrite,
            mode,
            columns,
        )
        return self

    def with_units(self, convention=None, conversions=None, **columns):
        self.call = ("with_units", convention, conversions, columns)
        return self


@pytest.fixture
def halo_properties(test_data):
    return oc.open(test_data.snapshot.primary.halo_properties)


def test_apply_filter_message():
    dataset = RecordingDataset()
    message = FilterMessage(
        masks=(
            ComparisonMask(
                operator="greater_than",
                left=ColumnReference(name="mass"),
                right=NumberLiteral(value=1),
            ),
        ),
        mode="local",
    )

    result = apply_message(dataset, message)  # type: ignore[arg-type]

    assert result is dataset
    assert dataset.call is not None
    assert dataset.call[0] == "filter"
    assert isinstance(dataset.call[1][0], ColumnMask)
    assert dataset.call[2] == "local"


def test_apply_select_message():
    dataset = RecordingDataset()
    message = SelectMessage(
        columns=("tag",),
        derived_columns={
            "scaled_mass": ArithmeticExpression(
                operator="multiply",
                left=ColumnReference(name="mass"),
                right=NumberLiteral(value=2),
            )
        },
    )

    result = apply_message(dataset, message)  # type: ignore[arg-type]

    assert result is dataset
    assert dataset.call is not None
    assert dataset.call[0:3] == ("select", ("tag",), "global")
    assert isinstance(dataset.call[3]["scaled_mass"], Column)


def test_apply_select_rejects_literal_derived_value():
    dataset = RecordingDataset()
    message = SelectMessage(derived_columns={"constant": NumberLiteral(value=1)})

    result = apply_message(dataset, message)  # type: ignore[arg-type]

    assert isinstance(result, SerdeError)
    assert "must depend on a column" in result.message


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (DropMessage(columns=("mass", "tag*")), ("drop", ("mass", "tag*"))),
        (SortByMessage(column="mass", invert=True), ("sort_by", "mass", True)),
        (SortByMessage(column=None), ("sort_by", None, False)),
        (TakeMessage(n=3, at="start"), ("take", 3, "start", "local")),
        (
            TakeRangeMessage(start=2, end=5, mode="global"),
            ("take_range", 2, 5, "global"),
        ),
    ],
)
def test_apply_structural_message(message, expected):
    dataset = RecordingDataset()

    assert apply_message(dataset, message) is dataset  # type: ignore[arg-type]
    assert dataset.call == expected


def test_apply_take_rows_message_builds_int64_array():
    dataset = RecordingDataset()

    assert apply_message(dataset, TakeRowsMessage(rows=(3, 1))) is dataset  # type: ignore[arg-type]
    assert dataset.call is not None
    assert dataset.call[0] == "take_rows"
    assert dataset.call[1].dtype == np.int64
    assert dataset.call[1].tolist() == [3, 1]


@pytest.mark.parametrize(
    ("message_type", "kwargs"),
    [
        (TakeMessage, {"n": -1}),
        (TakeRangeMessage, {"start": 2, "end": 1}),
        (TakeRowsMessage, {"rows": (0, -1)}),
        (DropMessage, {"columns": ()}),
    ],
)
def test_row_and_drop_messages_reject_invalid_values(message_type, kwargs):
    with pytest.raises(ValidationError):
        message_type(**kwargs)


def test_apply_bound_messages():
    cases = (
        BoxRegionMessage(p1=(0, 0, 0), p2=(1, 2, 3)),
        ConeRegionMessage(center=(10, 20), radius=2.0),
        SkyboxRegionMessage(p1=(10, -5), p2=(20, 5)),
        HealpixRegionMessage(pixels=frozenset({4, 1}), nside=2),
    )

    for region_message in cases:
        dataset = RecordingDataset()
        result = apply_message(  # type: ignore[arg-type]
            dataset, BoundMessage(region=region_message, select_by="center")
        )
        assert result is dataset
        assert dataset.call is not None
        assert dataset.call[0] == "bound"
        assert dataset.call[2] == "center"


def test_apply_with_new_columns_message():
    dataset = RecordingDataset()
    message = WithNewColumnsMessage(
        columns={
            "scaled_mass": ArithmeticExpression(
                operator="multiply",
                left=ColumnReference(name="mass"),
                right=NumberLiteral(value=2),
            )
        },
        descriptions={"scaled_mass": "Twice the mass"},
        allow_overwrite=True,
        mode="local",
    )

    assert apply_message(dataset, message) is dataset  # type: ignore[arg-type]
    assert dataset.call is not None
    assert dataset.call[0:4] == (
        "with_new_columns",
        {"scaled_mass": "Twice the mass"},
        True,
        "local",
    )
    assert isinstance(dataset.call[4]["scaled_mass"], Column)


def test_with_new_columns_rejects_scalars_and_unknown_descriptions():
    dataset = RecordingDataset()
    result = apply_message(  # type: ignore[arg-type]
        dataset,
        WithNewColumnsMessage(
            columns={
                "mean_mass": ReductionExpression(
                    operator="mean", operand=ColumnReference(name="mass")
                )
            }
        ),
    )

    assert isinstance(result, SerdeError)
    assert "must produce a column" in result.message

    with pytest.raises(ValidationError, match="unknown columns"):
        WithNewColumnsMessage(
            columns={"mass2": ColumnReference(name="mass")},
            descriptions={"missing": "Not present"},
        )


def test_apply_with_units_message_normalizes_units():
    dataset = RecordingDataset()
    message = WithUnitsMessage(
        convention="physical",
        conversions={"M_sun": "kg"},
        columns={"distance": "kilometer"},
    )

    assert apply_message(dataset, message) is dataset  # type: ignore[arg-type]
    assert dataset.call == (
        "with_units",
        "physical",
        {u.solMass: u.kg},
        {"distance": u.km},
    )


def test_with_units_rejects_invalid_and_duplicate_units():
    with pytest.raises(ValidationError, match="Invalid Astropy unit"):
        WithUnitsMessage(columns={"distance": "not-a-unit"})

    with pytest.raises(ValidationError, match="Duplicate source unit"):
        WithUnitsMessage(conversions={"M_sun": "kg", "solMass": "g"})


def test_healpix_region_message_validates_and_sorts_pixels():
    region = HealpixRegionMessage(pixels=frozenset({3, 1, 2}), nside=2)

    assert region.model_dump(mode="json")["pixels"] == [1, 2, 3]
    with pytest.raises(ValidationError, match="positive power of two"):
        HealpixRegionMessage(pixels=frozenset(), nside=3)
    with pytest.raises(ValidationError, match="pixels must be less than"):
        HealpixRegionMessage(pixels=frozenset({48}), nside=2)


def test_live_expression_uses_expected_operator():
    live = expression_to_live(
        ArithmeticExpression(
            operator="multiply",
            left=ColumnReference(name="mass"),
            right=NumberLiteral(value=2),
        )
    )

    assert isinstance(live, Column)
    assert live.operation is operator.mul


def test_messages_transform_an_actual_dataset(halo_properties):
    mass = "fof_halo_mass"
    tag = "fof_halo_tag"
    filtered = apply_message(
        halo_properties,
        FilterMessage(
            masks=(
                ComparisonMask(
                    operator="greater_than",
                    left=ColumnReference(name=mass),
                    right=NumberLiteral(value=1.0),
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
        SelectMessage(
            columns=(tag, "mass2"),
        ),
    )
    result = apply_message(
        apply_message(selected, SortByMessage(column=tag, invert=True)),
        TakeMessage(n=2, at="start"),
    ).get_data()

    expected = (
        halo_properties.filter(oc.col(mass) > 1.0)
        .with_new_columns(mass2=oc.col(mass) * 2.0)
        .select(tag, "mass2")
        .sort_by(tag, invert=True)
        .take(2, at="start")
        .get_data()
    )

    assert np.array_equal(result[tag], expected[tag])
    assert np.array_equal(result["mass2"], expected["mass2"])
