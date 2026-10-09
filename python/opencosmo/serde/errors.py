"""Structured errors returned by public serialization boundaries."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from opencosmo.column.select import MissingColumnError
from opencosmo.units import UnitsError


class SerdeErrorCategory(StrEnum):
    """Stable categories for serialization and operation failures."""

    UNSUPPORTED_TYPE = "unsupported_type"
    VALIDATION_ERROR = "validation_error"
    MISSING_COLUMN = "missing_column"
    INVALID_UNITS = "invalid_units"
    INVALID_OPERATION = "invalid_operation"
    OPERATION_ERROR = "operation_error"
    INTERNAL_ERROR = "internal_error"


class SerdeFieldError(BaseModel):
    """One field-level validation failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    location: tuple[str | int, ...]
    error_type: str
    message: str


class SerdeError(BaseModel):
    """A structured failure from a public serde operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["error"] = "error"
    operation: Literal[
        "serialize_result",
        "apply_message",
        "decode_message",
        "serialize_column_expression",
        "serialize_column_mask",
    ]
    category: SerdeErrorCategory
    message: str
    exception_type: str
    input_type: str | None = None
    target_type: str | None = None
    message_type: str | None = None
    field_errors: tuple[SerdeFieldError, ...] = ()


def make_serde_error(
    operation: Literal[
        "serialize_result",
        "apply_message",
        "decode_message",
        "serialize_column_expression",
        "serialize_column_mask",
    ],
    error: Exception,
    *,
    input_type: str | None = None,
    target_type: str | None = None,
    message_type: str | None = None,
) -> SerdeError:
    """Convert an ordinary exception into a structured error without raising."""
    try:
        message = str(error)
    except Exception:
        message = f"{operation} failed"

    if isinstance(error, ValidationError):
        category = SerdeErrorCategory.VALIDATION_ERROR
        field_errors = tuple(
            SerdeFieldError(
                location=tuple(entry["loc"]),
                error_type=entry["type"],
                message=entry["msg"],
            )
            for entry in error.errors(include_url=False, include_context=False)
        )
    else:
        field_errors = ()
        if isinstance(error, MissingColumnError):
            category = SerdeErrorCategory.MISSING_COLUMN
        elif isinstance(error, UnitsError):
            category = SerdeErrorCategory.INVALID_UNITS
        elif isinstance(error, TypeError):
            category = SerdeErrorCategory.UNSUPPORTED_TYPE
        elif isinstance(error, ValueError | AttributeError | KeyError | IndexError):
            category = SerdeErrorCategory.INVALID_OPERATION
        elif isinstance(error, RuntimeError | RecursionError):
            category = SerdeErrorCategory.INTERNAL_ERROR
        else:
            category = SerdeErrorCategory.OPERATION_ERROR

    try:
        return SerdeError(
            operation=operation,
            category=category,
            message=message,
            exception_type=type(error).__name__,
            input_type=input_type,
            target_type=target_type,
            message_type=message_type,
            field_errors=field_errors,
        )
    except Exception:
        return SerdeError.model_construct(
            kind="error",
            operation=operation,
            category=SerdeErrorCategory.INTERNAL_ERROR,
            message=f"{operation} failed",
            exception_type=type(error).__name__,
            input_type=input_type,
            target_type=target_type,
            message_type=message_type,
            field_errors=(),
        )


__all__ = ["SerdeError", "SerdeErrorCategory", "SerdeFieldError"]
