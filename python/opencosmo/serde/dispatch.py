"""Entrypoints for applying incoming messages to OpenCosmo objects.

An execution environment receives a JSON object describing an operation and
holds the OpenCosmo object it applies to. :func:`parse_message` validates the
payload against the messages that object supports, :func:`apply_message`
performs a parsed message, and :func:`execute_message` does both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from opencosmo.collection.lightcone import Lightcone
from opencosmo.collection.structure import StructureCollection
from opencosmo.dataset import Dataset

from .messages.dataset import DATASET_MESSAGE_TYPES, DatasetMessage
from .messages.lightcone import LIGHTCONE_MESSAGE_TYPES, LightconeMessage
from .messages.structure_collection import (
    STRUCTURE_COLLECTION_MESSAGE_TYPES,
    StructureCollectionMessage,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import JsonValue

    from opencosmo import (
        HealpixMap,
        SimulationCollection,
    )

type MessageTarget = (
    Dataset | HealpixMap | Lightcone | StructureCollection | SimulationCollection
)
type Message = DatasetMessage | LightconeMessage | StructureCollectionMessage


class MessageError(BaseModel):
    """Why a message could not be parsed or applied.

    ``kind`` is ``"unsupported_target"`` when the object has no messages,
    ``"invalid_message"`` when the payload fails validation,
    ``"unsupported_message"`` when a parsed message belongs to another type,
    and ``"operation_failed"`` when applying the message raised. In the last
    case ``error_type`` is the exception class name and ``detail`` its message.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[
        "unsupported_target",
        "invalid_message",
        "unsupported_message",
        "operation_failed",
    ]
    target: str
    detail: str
    error_type: str | None = None


__DATASET_MESSAGES: TypeAdapter[DatasetMessage] = TypeAdapter(DatasetMessage)
__LIGHTCONE_MESSAGES: TypeAdapter[LightconeMessage] = TypeAdapter(LightconeMessage)
__STRUCTURE_MESSAGES: TypeAdapter[StructureCollectionMessage] = TypeAdapter(
    StructureCollectionMessage
)


def __unsupported_target(target: type[MessageTarget]) -> MessageError:
    name = target.__name__
    return MessageError(
        kind="unsupported_target",
        target=name,
        detail=f"{name} does not support messages",
    )


@overload
def __apply(
    message: DatasetMessage, target: Dataset, name: str
) -> Dataset | MessageError: ...


@overload
def __apply(
    message: LightconeMessage, target: Lightcone, name: str
) -> Lightcone | MessageError: ...


@overload
def __apply(
    message: StructureCollectionMessage, target: StructureCollection, name: str
) -> StructureCollection | MessageError: ...


def __apply(
    message: Message, target: MessageTarget, name: str
) -> MessageTarget | MessageError:
    try:
        # The match in apply_message guarantees message and target are paired
        return message.apply(target)  # ty: ignore[invalid-argument-type]
    except Exception as error:  # noqa: BLE001 - operations raise arbitrary user-facing errors
        return MessageError(
            kind="operation_failed",
            target=name,
            detail=str(error),
            error_type=type(error).__name__,
        )


@overload
def parse_message(
    target: type[HealpixMap | Lightcone | StructureCollection | SimulationCollection],
    payload: Mapping[str, JsonValue],
) -> MessageError: ...


@overload
def parse_message(
    target: type[Dataset], payload: Mapping[str, JsonValue]
) -> DatasetMessage | MessageError: ...


@overload
def parse_message(
    target: type[Lightcone], payload: Mapping[str, JsonValue]
) -> LightconeMessage | MessageError: ...


@overload
def parse_message(
    target: type[StructureCollection], payload: Mapping[str, JsonValue]
) -> StructureCollectionMessage | MessageError: ...


@overload
def parse_message(
    target: type[MessageTarget], payload: Mapping[str, JsonValue]
) -> Message | MessageError: ...


def parse_message(
    target: type[MessageTarget], payload: Mapping[str, JsonValue]
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
        case _ if target is Dataset:
            try:
                return __DATASET_MESSAGES.validate_python(payload)
            except ValidationError as error:
                return MessageError(
                    kind="invalid_message", target="Dataset", detail=str(error)
                )
        case _ if target is Lightcone:
            try:
                return __LIGHTCONE_MESSAGES.validate_python(payload)
            except ValidationError as error:
                return MessageError(
                    kind="invalid_message", target="Lightcone", detail=str(error)
                )
        case _ if target is StructureCollection:
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
    target: HealpixMap | Lightcone | StructureCollection | SimulationCollection,
    message: Message,
) -> MessageError: ...


@overload
def apply_message(target: Dataset, message: Message) -> Dataset | MessageError: ...


@overload
@overload
def apply_message(target: Lightcone, message: Message) -> Lightcone | MessageError: ...


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
                return __apply(message, target, "Dataset")
            return MessageError(
                kind="unsupported_message",
                target="Dataset",
                detail=f"{type(message).__name__} cannot be applied to a Dataset",
            )
        case Lightcone():
            if isinstance(message, LIGHTCONE_MESSAGE_TYPES):
                return __apply(message, target, "Lightcone")
            return MessageError(
                kind="unsupported_message",
                target="Lightcone",
                detail=f"{type(message).__name__} cannot be applied to a Lightcone",
            )
        case StructureCollection():
            if isinstance(message, STRUCTURE_COLLECTION_MESSAGE_TYPES):
                return __apply(message, target, "StructureCollection")
            return MessageError(
                kind="unsupported_message",
                target="StructureCollection",
                detail=f"{type(message).__name__} cannot be applied to a StructureCollection",
            )
    return __unsupported_target(type(target))


@overload
def execute_message(
    target: HealpixMap | Lightcone | StructureCollection | SimulationCollection,
    payload: Mapping[str, JsonValue],
) -> MessageError: ...


@overload
def execute_message(
    target: Dataset, payload: Mapping[str, JsonValue]
) -> Dataset | MessageError: ...


@overload
def execute_message(
    target: Lightcone, payload: Mapping[str, JsonValue]
) -> Lightcone | MessageError: ...


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
    message = parse_message(type(target), payload)
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
