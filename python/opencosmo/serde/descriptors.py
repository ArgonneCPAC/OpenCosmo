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

    ``allowed_messages`` maps each method name to the message models that can
    represent a call to it. Most methods have exactly one; collections that
    delegate to children of different types may have several. ``signatures``
    holds the call signature of each candidate, in the same order, and
    ``messages_by_kind`` finds a model from its wire ``kind``.
    """

    type_name: str
    summary_type: type[SummaryModel]
    allowed_messages: dict[str, tuple[type[ExpressionModel], ...]]
    signatures: dict[str, tuple[tuple[MessageParameter, ...], ...]]
    messages_by_kind: dict[str, type[ExpressionModel]]


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


def __descriptor(
    type_name: str, summary_type: type[SummaryModel], alias: object
) -> DataClassDescriptor:
    models = [
        model
        for model in __flatten(alias)
        if issubclass(model, ExpressionModel) and hasattr(model, "method")
    ]
    by_method: dict[str, list[type[ExpressionModel]]] = {}
    for model in models:
        by_method.setdefault(model.method, []).append(model)
    return DataClassDescriptor(
        type_name,
        summary_type,
        {method: tuple(group) for method, group in by_method.items()},
        {
            method: tuple(message_signature(model) for model in group)
            for method, group in by_method.items()
        },
        {str(model.model_fields["kind"].default): model for model in models},
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
