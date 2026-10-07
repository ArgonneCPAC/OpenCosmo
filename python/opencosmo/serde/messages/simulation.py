"""Validated messages for simulation-collection transformations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from ..expression import Expression, ExpressionModel  # noqa: TC001
from .common import ColumnName  # noqa: TC001
from .dataset import (
    BoundMessage,
    DropMessage,
    FilterMessage,
    SelectMessage,
    SortByMessage,
    TakeMessage,
    TakeRangeMessage,
    WithNewColumnsMessage,
    WithUnitsMessage,
)
from .structure import (
    DatasetPath,  # noqa: TC001
    StructureDropMessage,
    StructureFilterMessage,
    StructureSelectMessage,
    StructureWithNewColumnsMessage,
    StructureWithUnitsMessage,
)

type SimulationName = Annotated[str, Field(min_length=1, pattern=r"^[^.]+$")]


class MatchMessage(ExpressionModel):
    """A request to align simulations to a matching source."""

    kind: Literal["match"] = "match"
    source: SimulationName


class ClearMatchMessage(ExpressionModel):
    """A request to clear an active simulation match source."""

    kind: Literal["clear_match"] = "clear_match"


class SimulationWithNewColumnsMessage(ExpressionModel):
    """A request to add derived columns to selected simulations."""

    kind: Literal["simulation_with_new_columns"] = "simulation_with_new_columns"
    simulations: tuple[SimulationName, ...] | None = None
    dataset: DatasetPath | None = None
    columns: dict[ColumnName, Expression] = Field(min_length=1)
    descriptions: str | dict[ColumnName, str] = Field(default_factory=dict)
    allow_overwrite: bool = False

    @model_validator(mode="after")
    def validate_message(self) -> Self:
        """Validate targets and descriptions."""
        if self.simulations is not None:
            if not self.simulations:
                raise ValueError("simulations must not be empty")
            if len(set(self.simulations)) != len(self.simulations):
                raise ValueError("simulations must not contain duplicates")
        if isinstance(self.descriptions, dict):
            unknown = self.descriptions.keys() - self.columns.keys()
            if unknown:
                raise ValueError(
                    f"Descriptions provided for unknown columns {sorted(unknown)}"
                )
        return self


type SimulationMessage = Annotated[
    FilterMessage
    | StructureFilterMessage
    | SelectMessage
    | StructureSelectMessage
    | DropMessage
    | StructureDropMessage
    | SortByMessage
    | TakeMessage
    | TakeRangeMessage
    | BoundMessage
    | WithUnitsMessage
    | StructureWithUnitsMessage
    | WithNewColumnsMessage
    | StructureWithNewColumnsMessage
    | SimulationWithNewColumnsMessage
    | MatchMessage
    | ClearMatchMessage,
    Field(discriminator="kind"),
]


__all__ = [
    "ClearMatchMessage",
    "MatchMessage",
    "SimulationMessage",
    "SimulationName",
    "SimulationWithNewColumnsMessage",
]
