"""Descriptions of the messages and summaries available for each result type."""

from __future__ import annotations

from dataclasses import dataclass
from types import UnionType
from typing import TYPE_CHECKING, Annotated, TypeAliasType, Union, get_args, get_origin

from pydantic import BaseModel

from .expression import ExpressionModel
from .messages import (
    DatasetMessage,
    HealpixMapMessage,
    LightconeMessage,
    SimulationMessage,
    StructureMessage,
)
from .summary import (
    DatasetSummary,
    HealpixMapSummary,
    LightconeSummary,
    SimulationCollectionSummary,
    StructureCollectionSummary,
)

if TYPE_CHECKING:
    from .summary import SummaryModel


@dataclass(frozen=True)
class DataClassDescriptor:
    """Everything a consumer needs to build a proxy for one result type.

    ``allowed_messages`` maps each message ``kind`` to its model class.
    """

    type_name: str
    summary_type: type[SummaryModel]
    allowed_messages: dict[str, type[ExpressionModel]]


def __flatten(annotation: object) -> list[type[BaseModel]]:
    if isinstance(annotation, TypeAliasType):
        return __flatten(annotation.__value__)
    origin = get_origin(annotation)
    if origin is Annotated:
        return __flatten(get_args(annotation)[0])
    if origin in (Union, UnionType):
        return [member for arg in get_args(annotation) for member in __flatten(arg)]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    raise TypeError(f"Cannot extract message models from {annotation!r}")


def __messages(alias: object) -> dict[str, type[ExpressionModel]]:
    return {
        str(model.model_fields["kind"].default): model
        for model in __flatten(alias)
        if issubclass(model, ExpressionModel)
    }


DESCRIPTORS: dict[str, DataClassDescriptor] = {
    descriptor.type_name: descriptor
    for descriptor in (
        DataClassDescriptor("Dataset", DatasetSummary, __messages(DatasetMessage)),
        DataClassDescriptor(
            "Lightcone", LightconeSummary, __messages(LightconeMessage)
        ),
        DataClassDescriptor(
            "HealpixMap", HealpixMapSummary, __messages(HealpixMapMessage)
        ),
        DataClassDescriptor(
            "StructureCollection",
            StructureCollectionSummary,
            __messages(StructureMessage),
        ),
        DataClassDescriptor(
            "SimulationCollection",
            SimulationCollectionSummary,
            __messages(SimulationMessage),
        ),
    )
}

__all__ = ["DESCRIPTORS", "DataClassDescriptor"]
