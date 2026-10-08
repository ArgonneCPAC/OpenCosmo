"""Markers describing how message fields bind to method parameters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from .expression import ExpressionModel


@dataclass(frozen=True)
class VarArgs:
    """Marks the field that collects a method's ``*args``."""


@dataclass(frozen=True)
class VarKwargs:
    """Marks the field that collects a method's ``**kwargs``."""


@dataclass(frozen=True)
class KeywordOnly:
    """Marks a field that a method accepts only by keyword."""


class ParameterRole(StrEnum):
    """How a message field is bound when a proxy parses a call."""

    POSITIONAL = "positional"
    VAR_ARGS = "var_args"
    KEYWORD_ONLY = "keyword_only"
    VAR_KWARGS = "var_kwargs"


class MessageParameter(NamedTuple):
    """One message field and the role it plays in the corresponding call."""

    name: str
    role: ParameterRole


def message_signature(model: type[ExpressionModel]) -> tuple[MessageParameter, ...]:
    """Describe a message model as an ordered call signature.

    Fields appear in declaration order, which must match the order of the
    parameters of the method the message represents. ``kind`` is excluded.
    """
    parameters: list[MessageParameter] = []
    for name, field in model.model_fields.items():
        if name == "kind":
            continue
        role = ParameterRole.POSITIONAL
        for marker in field.metadata:
            if isinstance(marker, VarArgs):
                role = ParameterRole.VAR_ARGS
            elif isinstance(marker, VarKwargs):
                role = ParameterRole.VAR_KWARGS
            elif isinstance(marker, KeywordOnly):
                role = ParameterRole.KEYWORD_ONLY
        parameters.append(MessageParameter(name, role))
    return tuple(parameters)


__all__ = [
    "KeywordOnly",
    "MessageParameter",
    "ParameterRole",
    "VarArgs",
    "VarKwargs",
    "message_signature",
]
