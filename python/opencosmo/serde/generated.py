"""Message sets generated from the public API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from opencosmo.dataset import Dataset

from .generate import generate_messages

if TYPE_CHECKING:
    from pydantic import BaseModel


def __ordered_range[M: BaseModel](message: M) -> M:
    if message.end < message.start:  # type: ignore[attr-defined]
        raise ValueError("end must be greater than or equal to start")
    return message


def __described_columns[M: BaseModel](message: M) -> M:
    descriptions = message.descriptions  # type: ignore[attr-defined]
    if isinstance(descriptions, dict):
        unknown = descriptions.keys() - message.new_columns.keys()  # type: ignore[attr-defined]
        if unknown:
            raise ValueError(
                f"Descriptions provided for unknown columns {sorted(unknown)}"
            )
    return message


DATASET_MESSAGES = generate_messages(
    Dataset,
    (
        "filter",
        "select",
        "drop",
        "sort_by",
        "take",
        "take_range",
        "take_rows",
        "bound",
        "with_new_columns",
        "with_units",
    ),
    validators={
        "take_range": (__ordered_range,),
        "with_new_columns": (__described_columns,),
    },
)

__all__ = ["DATASET_MESSAGES"]
