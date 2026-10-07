"""Validated messages for dataset transformations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from opencosmo.units import UnitConvention  # noqa: TC001

from ..expression import Expression, ExpressionModel, Mask  # noqa: TC001
from .common import (
    ColumnName,  # noqa: TC001
    NonNegativeInt,  # noqa: TC001
    ReductionMode,
    TakePosition,
    normalize_unit,
)
from .region import RegionMessage  # noqa: TC001


class FilterMessage(ExpressionModel):
    """A request to filter a dataset with one or more masks."""

    kind: Literal["filter"] = "filter"
    masks: tuple[Mask, ...] = ()
    mode: ReductionMode = ReductionMode.GLOBAL


class SelectMessage(ExpressionModel):
    """A request to select existing and derived dataset columns."""

    kind: Literal["select"] = "select"
    columns: tuple[str, ...] = ()
    derived_columns: dict[str, Expression] = Field(default_factory=dict)
    mode: ReductionMode = ReductionMode.GLOBAL

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Reject empty column names and patterns."""
        if any(not value for value in values):
            raise ValueError("Column names and patterns must not be empty")
        return values

    @field_validator("derived_columns")
    @classmethod
    def validate_derived_columns(
        cls, values: dict[str, Expression]
    ) -> dict[str, Expression]:
        """Reject empty output column names."""
        if any(not name for name in values):
            raise ValueError("Derived column names must not be empty")
        return values


class DropMessage(ExpressionModel):
    """A request to remove visible dataset columns or wildcard matches."""

    kind: Literal["drop"] = "drop"
    columns: tuple[ColumnName, ...] = Field(min_length=1)


class SortByMessage(ExpressionModel):
    """A request to sort a dataset or clear its current sorting."""

    kind: Literal["sort_by"] = "sort_by"
    column: ColumnName | None = None
    invert: bool = False


class TakeMessage(ExpressionModel):
    """A request to select a number of rows from a dataset."""

    kind: Literal["take"] = "take"
    n: NonNegativeInt
    at: TakePosition = TakePosition.RANDOM
    mode: ReductionMode = ReductionMode.LOCAL


class TakeRangeMessage(ExpressionModel):
    """A request to select a half-open range of dataset rows."""

    kind: Literal["take_range"] = "take_range"
    start: NonNegativeInt
    end: NonNegativeInt
    mode: ReductionMode = ReductionMode.LOCAL

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        """Require the end of the range to follow its start."""
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


class TakeRowsMessage(ExpressionModel):
    """A request to select explicit dataset row positions."""

    kind: Literal["take_rows"] = "take_rows"
    rows: tuple[NonNegativeInt, ...]


class BoundMessage(ExpressionModel):
    """A request to spatially bound a dataset."""

    kind: Literal["bound"] = "bound"
    region: RegionMessage
    select_by: ColumnName | None = None


class WithNewColumnsMessage(ExpressionModel):
    """A request to add expression-derived columns to a dataset."""

    kind: Literal["with_new_columns"] = "with_new_columns"
    columns: dict[ColumnName, Expression] = Field(min_length=1)
    descriptions: str | dict[ColumnName, str] = Field(default_factory=dict)
    allow_overwrite: bool = False
    mode: ReductionMode = ReductionMode.GLOBAL

    @model_validator(mode="after")
    def validate_descriptions(self) -> Self:
        """Require mapped descriptions to describe only supplied columns."""
        if isinstance(self.descriptions, dict):
            unknown = self.descriptions.keys() - self.columns.keys()
            if unknown:
                raise ValueError(
                    f"Descriptions provided for unknown columns {sorted(unknown)}"
                )
        return self


class WithUnitsMessage(ExpressionModel):
    """A request to change unit convention or apply unit conversions."""

    kind: Literal["with_units"] = "with_units"
    convention: UnitConvention | None = None
    conversions: dict[str, str] = Field(default_factory=dict)
    columns: dict[ColumnName, str] = Field(default_factory=dict)

    @field_validator("conversions")
    @classmethod
    def validate_conversions(cls, values: dict[str, str]) -> dict[str, str]:
        """Validate and normalize blanket unit conversions."""
        normalized: dict[str, str] = {}
        for source, target in values.items():
            source_unit = normalize_unit(source)
            if source_unit in normalized:
                raise ValueError(
                    f"Duplicate source unit after normalization: {source_unit}"
                )
            normalized[source_unit] = normalize_unit(target)
        return normalized

    @field_validator("columns")
    @classmethod
    def validate_column_units(cls, values: dict[str, str]) -> dict[str, str]:
        """Validate and normalize per-column target units."""
        return {name: normalize_unit(unit) for name, unit in values.items()}


type DatasetMessageType = (
    FilterMessage
    | SelectMessage
    | DropMessage
    | SortByMessage
    | TakeMessage
    | TakeRangeMessage
    | TakeRowsMessage
    | BoundMessage
    | WithNewColumnsMessage
    | WithUnitsMessage
)

type DatasetMessage = Annotated[DatasetMessageType, Field(discriminator="kind")]


__all__ = [
    "BoundMessage",
    "DatasetMessage",
    "DropMessage",
    "FilterMessage",
    "SelectMessage",
    "SortByMessage",
    "TakeMessage",
    "TakeRangeMessage",
    "TakeRowsMessage",
    "WithNewColumnsMessage",
    "WithUnitsMessage",
]
