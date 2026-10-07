"""Validated messages for dataset operations."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

import astropy.units as u
import healpy as hp
from pydantic import (
    Field,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from opencosmo.units import UnitConvention  # noqa: TC001

from .expression import Expression, ExpressionModel, FiniteNumber, Mask  # noqa: TC001

type ColumnName = Annotated[str, Field(min_length=1)]
type NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
type PositiveInt = Annotated[StrictInt, Field(gt=0)]


class ReductionMode(StrEnum):
    """Scope used to compute scalar reductions."""

    LOCAL = "local"
    GLOBAL = "global"


class TakePosition(StrEnum):
    """Position from which rows are selected."""

    START = "start"
    END = "end"
    RANDOM = "random"


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


class BoxRegionMessage(ExpressionModel):
    """A three-dimensional box query."""

    kind: Literal["box"] = "box"
    p1: tuple[FiniteNumber, FiniteNumber, FiniteNumber]
    p2: tuple[FiniteNumber, FiniteNumber, FiniteNumber]


class ConeRegionMessage(ExpressionModel):
    """A circular sky query in degrees."""

    kind: Literal["cone"] = "cone"
    center: tuple[FiniteNumber, FiniteNumber]
    radius: Annotated[FiniteNumber, Field(gt=0)]


class SkyboxRegionMessage(ExpressionModel):
    """A rectangular sky query in degrees."""

    kind: Literal["skybox"] = "skybox"
    p1: tuple[FiniteNumber, FiniteNumber]
    p2: tuple[FiniteNumber, FiniteNumber]


class HealpixRegionMessage(ExpressionModel):
    """A HEALPix query containing explicit nested pixel identifiers."""

    kind: Literal["healpix"] = "healpix"
    pixels: frozenset[NonNegativeInt]
    nside: PositiveInt

    @model_validator(mode="after")
    def validate_healpix(self) -> Self:
        """Validate the resolution and pixel range."""
        if not hp.isnsideok(self.nside) or self.nside & (self.nside - 1):
            raise ValueError("nside must be a positive power of two")
        pixel_count = hp.nside2npix(self.nside)
        if any(pixel >= pixel_count for pixel in self.pixels):
            raise ValueError(f"pixels must be less than {pixel_count} for this nside")
        return self

    @field_serializer("pixels")
    def serialize_pixels(self, pixels: frozenset[int]) -> list[int]:
        """Serialize pixel identifiers in deterministic order."""
        return sorted(pixels)


type RegionMessage = Annotated[
    BoxRegionMessage | ConeRegionMessage | SkyboxRegionMessage | HealpixRegionMessage,
    Field(discriminator="kind"),
]


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
            source_unit = _normalize_unit(source)
            if source_unit in normalized:
                raise ValueError(
                    f"Duplicate source unit after normalization: {source_unit}"
                )
            normalized[source_unit] = _normalize_unit(target)
        return normalized

    @field_validator("columns")
    @classmethod
    def validate_column_units(cls, values: dict[str, str]) -> dict[str, str]:
        """Validate and normalize per-column target units."""
        return {name: _normalize_unit(unit) for name, unit in values.items()}


type DatasetMessage = Annotated[
    FilterMessage
    | SelectMessage
    | DropMessage
    | SortByMessage
    | TakeMessage
    | TakeRangeMessage
    | TakeRowsMessage
    | BoundMessage
    | WithNewColumnsMessage
    | WithUnitsMessage,
    Field(discriminator="kind"),
]


def _normalize_unit(value: str) -> str:
    if not value:
        raise ValueError("Unit strings must not be empty")
    try:
        return u.Unit(value).to_string()
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid Astropy unit: {value!r}") from error


__all__ = [
    "BoundMessage",
    "BoxRegionMessage",
    "ColumnName",
    "ConeRegionMessage",
    "DatasetMessage",
    "DropMessage",
    "FilterMessage",
    "HealpixRegionMessage",
    "NonNegativeInt",
    "PositiveInt",
    "RegionMessage",
    "ReductionMode",
    "SelectMessage",
    "SkyboxRegionMessage",
    "SortByMessage",
    "TakeMessage",
    "TakePosition",
    "TakeRangeMessage",
    "TakeRowsMessage",
    "WithNewColumnsMessage",
    "WithUnitsMessage",
]
