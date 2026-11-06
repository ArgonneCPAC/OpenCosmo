"""Semantic message checks that cannot be inferred from method signatures.

Each check receives the validated values of the message fields it is configured
with in :mod:`opencosmo.serde.codegen.config` and raises ``ValueError`` when the
combination is invalid.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from .common import validate_healpix

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping


def explicit_parameters(method: Callable[..., object]) -> tuple[str, ...]:
    """Return the named parameters of a method, excluding its receiver and variadics."""
    return tuple(
        parameter.name
        for parameter in list(inspect.signature(method).parameters.values())[1:]
        if parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    )


def reserved_keywords(keywords: Collection[str], reserved: Collection[str]) -> None:
    """Reject variadic keywords that collide with explicit method parameters."""
    if collisions := set(keywords).intersection(reserved):
        raise ValueError(f"Reserved keyword names: {sorted(collisions)}")


def ordered_range(start: int, end: int) -> None:
    """Require the end of a half-open range to follow its start."""
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    if start < 0:
        raise ValueError("start of range cannot be less than 0")


def described_columns(
    descriptions: str | Mapping[str, str] | None, columns: Collection[str]
) -> None:
    """Require mapped descriptions to describe only supplied columns."""
    if descriptions is None or isinstance(descriptions, str):
        return
    if unknown := descriptions.keys() - set(columns):
        raise ValueError(f"Descriptions provided for unknown columns {sorted(unknown)}")


def pixel_search(pixels: Collection[int], nside: int) -> None:
    """Require nonempty, valid nested HEALPix query pixels."""
    validate_healpix(pixels, nside)
    if not pixels:
        raise ValueError("Pixels must be a 1d array of positive integers")


__all__ = [
    "described_columns",
    "explicit_parameters",
    "ordered_range",
    "pixel_search",
    "reserved_keywords",
]
