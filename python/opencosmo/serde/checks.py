"""Semantic message checks that cannot be inferred from method signatures.

Each check receives the validated values of the message fields it is configured
with in :mod:`opencosmo.serde.codegen.config` and raises ``ValueError`` when the
combination is invalid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping


def ordered_range(start: int, end: int) -> None:
    """Require the end of a half-open range to follow its start."""
    if end < start:
        raise ValueError("end must be greater than or equal to start")


def described_columns(
    descriptions: str | Mapping[str, str] | None, columns: Collection[str]
) -> None:
    """Require mapped descriptions to describe only supplied columns."""
    if descriptions is None or isinstance(descriptions, str):
        return
    if unknown := descriptions.keys() - set(columns):
        raise ValueError(f"Descriptions provided for unknown columns {sorted(unknown)}")


__all__ = ["described_columns", "ordered_range"]
