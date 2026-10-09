"""Decode incoming payloads into validated messages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_core import from_json

from .descriptors import DESCRIPTORS
from .errors import make_serde_error

if TYPE_CHECKING:
    from .errors import SerdeError
    from .expression import ExpressionModel


def __decode(target_type: str, payload: bytes | str | dict[str, object]):
    descriptor = DESCRIPTORS.get(target_type)
    if descriptor is None:
        raise ValueError(
            f"Unknown target type {target_type!r}, expected one of {sorted(DESCRIPTORS)}"
        )
    data = from_json(payload) if isinstance(payload, bytes | str) else payload
    if not isinstance(data, dict):
        raise TypeError(f"Message must be an object, got {type(data).__name__}")
    kind = data.get("kind")
    if not isinstance(kind, str):
        raise ValueError("Message is missing a string 'kind'")
    model = descriptor.messages_by_kind.get(kind)
    if model is None:
        raise ValueError(f"Message kind {kind!r} is not supported for {target_type}")
    return model.model_validate(data)


def decode_message(
    target_type: str, payload: bytes | str | dict[str, object]
) -> ExpressionModel | SerdeError:
    """Decode a payload into the message model for ``target_type``.

    Parameters
    ----------
    target_type : str
        A key of ``DESCRIPTORS``, such as ``"Dataset"``.
    payload : bytes, str, or dict
        The message as JSON, or already parsed.

    Returns
    -------
    ExpressionModel or SerdeError
        The validated message, or a structured error describing why decoding failed.
    """
    try:
        return __decode(target_type, payload)
    except Exception as error:
        return make_serde_error(
            "decode_message",
            error,
            target_type=target_type,
            input_type=type(payload).__name__,
        )


__all__ = ["decode_message"]
