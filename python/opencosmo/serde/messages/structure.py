"""Validated messages for structure-collection transformations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from opencosmo.units import UnitConvention  # noqa: TC001

from ..expression import Expression, ExpressionModel, Mask  # noqa: TC001
from .common import (
    ColumnName,  # noqa: TC001
    ReductionMode,
    normalize_unit,
)
from .dataset import (
    BoundMessage,
    SortByMessage,
    TakeMessage,
    TakeRangeMessage,
    TakeRowsMessage,
)
from .lightcone import PixelSearchMessage, WithRedshiftRangeMessage

type DatasetPath = Annotated[str, Field(min_length=1, pattern=r"^[^.]+(?:\.[^.]+)*$")]


class StructureFilterMessage(ExpressionModel):
    """A request to filter structures by source or galaxy properties."""

    kind: Literal["structure_filter"] = "structure_filter"
    masks: tuple[Mask, ...] = ()
    on_galaxies: bool = False
    mode: ReductionMode = ReductionMode.GLOBAL


class StructureSelectionTarget(ExpressionModel):
    """Column selections for one dataset in a structure collection."""

    columns: tuple[ColumnName, ...] = ()
    derived_columns: dict[ColumnName, Expression] = Field(default_factory=dict)


class StructureSelectMessage(ExpressionModel):
    """A request to select columns across a structure collection."""

    kind: Literal["structure_select"] = "structure_select"
    columns: tuple[ColumnName, ...] = ()
    derived_columns: dict[ColumnName, Expression] = Field(default_factory=dict)
    targets: dict[DatasetPath, StructureSelectionTarget] = Field(default_factory=dict)
    mode: ReductionMode = ReductionMode.GLOBAL

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Keep automatic and explicitly targeted selection unambiguous."""
        if self.targets and (self.columns or self.derived_columns):
            raise ValueError(
                "Automatic columns and derived_columns cannot be combined with targets"
            )
        return self


class StructureDropTarget(ExpressionModel):
    """Columns to remove from one dataset in a structure collection."""

    columns: tuple[ColumnName, ...] = Field(min_length=1)


class StructureDropMessage(ExpressionModel):
    """A request to drop automatically routed or explicitly targeted columns."""

    kind: Literal["structure_drop"] = "structure_drop"
    columns: tuple[ColumnName, ...] = ()
    targets: dict[DatasetPath, StructureDropTarget] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Require at least one automatic or targeted drop."""
        if not self.columns and not self.targets:
            raise ValueError("At least one column or target must be provided")
        return self


class StructureWithNewColumnsMessage(ExpressionModel):
    """A request to add derived columns to a targeted dataset."""

    kind: Literal["structure_with_new_columns"] = "structure_with_new_columns"
    dataset: DatasetPath
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


class StructureUnitTarget(ExpressionModel):
    """Unit conversions for one dataset in a structure collection."""

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


class StructureWithUnitsMessage(ExpressionModel):
    """A request to apply collection-wide and per-dataset unit conversions."""

    kind: Literal["structure_with_units"] = "structure_with_units"
    convention: UnitConvention | None = None
    conversions: dict[str, str] = Field(default_factory=dict)
    datasets: dict[DatasetPath, StructureUnitTarget] = Field(default_factory=dict)

    @field_validator("conversions")
    @classmethod
    def validate_conversions(cls, values: dict[str, str]) -> dict[str, str]:
        """Validate and normalize collection-wide unit conversions."""
        normalized: dict[str, str] = {}
        for source, target in values.items():
            source_unit = normalize_unit(source)
            if source_unit in normalized:
                raise ValueError(
                    f"Duplicate source unit after normalization: {source_unit}"
                )
            normalized[source_unit] = normalize_unit(target)
        return normalized

    @field_validator("datasets")
    @classmethod
    def validate_datasets(
        cls, values: dict[str, StructureUnitTarget]
    ) -> dict[str, StructureUnitTarget]:
        """Restrict unit targets to direct collection members."""
        if any("." in name for name in values):
            raise ValueError("Unit conversion targets must be direct dataset names")
        return values


class WithDatasetsMessage(ExpressionModel):
    """A request to retain selected datasets in a structure collection."""

    kind: Literal["with_datasets"] = "with_datasets"
    datasets: tuple[DatasetPath, ...] = Field(min_length=1)


type StructureMessage = Annotated[
    StructureFilterMessage
    | StructureSelectMessage
    | StructureDropMessage
    | SortByMessage
    | TakeMessage
    | TakeRangeMessage
    | TakeRowsMessage
    | BoundMessage
    | StructureWithNewColumnsMessage
    | StructureWithUnitsMessage
    | WithDatasetsMessage
    | WithRedshiftRangeMessage
    | PixelSearchMessage,
    Field(discriminator="kind"),
]


__all__ = [
    "DatasetPath",
    "StructureDropMessage",
    "StructureDropTarget",
    "StructureFilterMessage",
    "StructureMessage",
    "StructureSelectMessage",
    "StructureSelectionTarget",
    "StructureUnitTarget",
    "StructureWithNewColumnsMessage",
    "StructureWithUnitsMessage",
    "WithDatasetsMessage",
]
