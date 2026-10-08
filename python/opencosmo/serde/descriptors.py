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
from .params import message_signature
from .summary import (
    DatasetSummary,
    HealpixMapSummary,
    LightconeSummary,
    SimulationCollectionSummary,
    StructureCollectionSummary,
)

if TYPE_CHECKING:
    from .params import MessageParameter
    from .summary import SummaryModel


@dataclass(frozen=True)
class DataClassDescriptor:
    """Everything a consumer needs to build a proxy for one result type.

    ``allowed_messages`` maps each message ``kind`` to its model class, and
    ``signatures`` maps it to the ordered call signature of that message.
    """

    type_name: str
    summary_type: type[SummaryModel]
    allowed_messages: dict[str, type[ExpressionModel]]
    signatures: dict[str, tuple[MessageParameter, ...]]


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


def __descriptor(
    type_name: str, summary_type: type[SummaryModel], alias: object
) -> DataClassDescriptor:
    messages = __messages(alias)
    return DataClassDescriptor(
        type_name,
        summary_type,
        messages,
        {kind: message_signature(model) for kind, model in messages.items()},
    )


DESCRIPTORS: dict[str, DataClassDescriptor] = {
    descriptor.type_name: descriptor
    for descriptor in (
        __descriptor("Dataset", DatasetSummary, DatasetMessage),
        __descriptor("Lightcone", LightconeSummary, LightconeMessage),
        __descriptor("HealpixMap", HealpixMapSummary, HealpixMapMessage),
        __descriptor(
            "StructureCollection", StructureCollectionSummary, StructureMessage
        ),
        __descriptor(
            "SimulationCollection", SimulationCollectionSummary, SimulationMessage
        ),
    )
}

__all__ = ["DESCRIPTORS", "DataClassDescriptor"]
