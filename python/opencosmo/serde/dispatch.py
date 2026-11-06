"""Entrypoints for applying incoming messages to OpenCosmo objects.

An execution environment receives a JSON object describing an operation and
holds the OpenCosmo object it applies to. :func:`parse_message` validates the
payload against the messages that object supports, :func:`apply_message`
performs a parsed message, and :func:`execute_message` does both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from opencosmo.collection.structure import StructureCollection
from opencosmo.dataset import Dataset

from .messages.dataset import DATASET_MESSAGE_TYPES, DatasetMessage
from .messages.structure_collection import (
    STRUCTURE_COLLECTION_MESSAGE_TYPES,
    StructureCollectionMessage,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import JsonValue

    from opencosmo import (
        HealpixMap,
        Lightcone,
        SimulationCollection,
    )

type MessageTarget = (
    Dataset | HealpixMap | Lightcone | StructureCollection | SimulationCollection
)
type Message = DatasetMessage | StructureCollectionMessage


class MessageError(BaseModel):
    """Why a message could not be parsed or applied.

    ``kind`` is ``"unsupported_target"`` when the object has no messages,
    ``"invalid_message"`` when the payload fails validation, and
    ``"unsupported_message"`` when a parsed message belongs to another type.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["unsupported_target", "invalid_message", "unsupported_message"]
    target: str
    detail: str


__DATASET_MESSAGES: TypeAdapter[DatasetMessage] = TypeAdapter(DatasetMessage)
__STRUCTURE_MESSAGES: TypeAdapter[StructureCollectionMessage] = TypeAdapter(
    StructureCollectionMessage
)


def __unsupported_target(target: MessageTarget) -> MessageError:
    name = type(target).__name__
    return MessageError(
        kind="unsupported_target",
        target=name,
        detail=f"{name} does not support messages",
    )


@overload
def parse_message(
    target: HealpixMap | Lightcone | SimulationCollection,
    payload: Mapping[str, JsonValue],
) -> MessageError: ...


@overload
def parse_message(
    target: Dataset, payload: Mapping[str, JsonValue]
) -> DatasetMessage | MessageError: ...


@overload
def parse_message(
    target: StructureCollection, payload: Mapping[str, JsonValue]
) -> StructureCollectionMessage | MessageError: ...


@overload
def parse_message(
    target: MessageTarget, payload: Mapping[str, JsonValue]
) -> Message | MessageError: ...


def parse_message(
    target: MessageTarget, payload: Mapping[str, JsonValue]
) -> Message | MessageError:
    """Validate a decoded JSON object as a message for ``target``.

    Parameters
    ----------
    target : Dataset or collection
        The object the message will be applied to. Its type selects the
        messages the payload is validated against.
    payload : mapping
        The decoded JSON message, discriminated by its ``kind``.

    Returns
    -------
    Message or MessageError
        The validated message, or why it could not be parsed.
    """
    match target:
        case Dataset():
            try:
                return __DATASET_MESSAGES.validate_python(payload)
            except ValidationError as error:
                return MessageError(
                    kind="invalid_message", target="Dataset", detail=str(error)
                )
        case StructureCollection():
            try:
                return __STRUCTURE_MESSAGES.validate_python(payload)
            except ValidationError as error:
                return MessageError(
                    kind="invalid_message",
                    target="StructureCollection",
                    detail=str(error),
                )
    return __unsupported_target(target)


@overload
def apply_message(
    target: HealpixMap | Lightcone | SimulationCollection, message: Message
) -> MessageError: ...


@overload
def apply_message(target: Dataset, message: Message) -> Dataset | MessageError: ...


@overload
def apply_message(
    target: StructureCollection, message: Message
) -> StructureCollection | MessageError: ...


@overload
def apply_message(
    target: MessageTarget, message: Message
) -> MessageTarget | MessageError: ...


def apply_message(
    target: MessageTarget, message: Message
) -> MessageTarget | MessageError:
    """Apply a parsed message to ``target``.

    Parameters
    ----------
    target : Dataset or collection
        The object to transform.
    message : Message
        A message returned by :func:`parse_message`.

    Returns
    -------
    Dataset, collection, or MessageError
        The transformed object, or why the message does not apply to it.
    """
    match target:
        case Dataset():
            if isinstance(message, DATASET_MESSAGE_TYPES):
                return message.apply(target)
            return MessageError(
                kind="unsupported_message",
                target="Dataset",
                detail=f"{type(message).__name__} cannot be applied to a Dataset",
            )
        case StructureCollection():
            if isinstance(message, STRUCTURE_COLLECTION_MESSAGE_TYPES):
                return message.apply(target)
            return MessageError(
                kind="unsupported_message",
                target="StructureCollection",
                detail=f"{type(message).__name__} cannot be applied to a StructureCollection",
            )
    return __unsupported_target(target)


@overload
def execute_message(
    target: HealpixMap | Lightcone | SimulationCollection,
    payload: Mapping[str, JsonValue],
) -> MessageError: ...


@overload
def execute_message(
    target: Dataset, payload: Mapping[str, JsonValue]
) -> Dataset | MessageError: ...


@overload
def execute_message(
    target: StructureCollection, payload: Mapping[str, JsonValue]
) -> StructureCollection | MessageError: ...


@overload
def execute_message(
    target: MessageTarget, payload: Mapping[str, JsonValue]
) -> MessageTarget | MessageError: ...


def execute_message(
    target: MessageTarget, payload: Mapping[str, JsonValue]
) -> MessageTarget | MessageError:
    """Parse a decoded JSON object as a message and apply it to ``target``.

    Parameters
    ----------
    target : Dataset or collection
        The object to transform.
    payload : mapping
        The decoded JSON message, discriminated by its ``kind``.

    Returns
    -------
    Dataset, collection, or MessageError
        The transformed object, or why the message could not be parsed or
        applied.
    """
    message = parse_message(target, payload)
    if isinstance(message, MessageError):
        return message
    return apply_message(target, message)


__all__ = [
    "Message",
    "MessageError",
    "MessageTarget",
    "apply_message",
    "execute_message",
    "parse_message",
]
