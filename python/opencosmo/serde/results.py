"""Result messages: server-to-client summaries of OpenCosmo objects.

Request messages in :mod:`opencosmo.serde.messages` are generated from public
method signatures. Result messages are hand-written: after an operation is
applied, the server sends one back describing the resultant object. Only
:class:`~opencosmo.Dataset`, :class:`~opencosmo.Lightcone`, and
:class:`~opencosmo.StructureCollection` are supported.
"""

from __future__ import annotations

from typing import Annotated, Literal, overload

from pydantic import BaseModel, ConfigDict, Field

from opencosmo.collection.lightcone import Lightcone
from opencosmo.collection.structure import StructureCollection
from opencosmo.dataset import Dataset

from .common import Keyword, NonNegativeInt


class ColumnSummary(BaseModel):
    """One column's name, description, and unit as seen by a client."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Keyword
    description: str | None
    unit: str | None


class DatasetSummary(BaseModel):
    """Summary of a resultant :class:`opencosmo.Dataset`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["dataset"] = "dataset"
    length: NonNegativeInt
    columns: tuple[ColumnSummary, ...]

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> DatasetSummary:
        """Summarize ``dataset``."""
        return cls(length=len(dataset), columns=_column_summaries(dataset))


class LightconeSummary(BaseModel):
    """Summary of a resultant :class:`opencosmo.Lightcone` and its children."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["lightcone"] = "lightcone"
    length: NonNegativeInt
    columns: tuple[ColumnSummary, ...]
    datasets: dict[str, DatasetSummary | LightconeSummary]

    @classmethod
    def from_lightcone(cls, lightcone: Lightcone) -> LightconeSummary:
        """Summarize ``lightcone``, recursing into each redshift step."""
        return cls(
            length=len(lightcone),
            columns=_column_summaries(lightcone),
            datasets={
                str(key): _summarize_child(child) for key, child in lightcone.items()
            },
        )


class StructureCollectionSummary(BaseModel):
    """Summary of a resultant :class:`opencosmo.StructureCollection`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["structure_collection"] = "structure_collection"
    length: NonNegativeInt
    dtype: str
    datasets: dict[str, DatasetSummary | StructureCollectionSummary]

    @classmethod
    def from_collection(
        cls, collection: StructureCollection
    ) -> StructureCollectionSummary:
        """Summarize ``collection``, recursing into each linked dataset."""
        return cls(
            length=len(collection),
            dtype=collection.dtype,
            datasets={
                str(key): _summarize_child(child) for key, child in collection.items()
            },
        )


type ResultMessage = Annotated[
    DatasetSummary | LightconeSummary | StructureCollectionSummary,
    Field(discriminator="kind"),
]
"""A summary of any supported resultant object, discriminated by ``kind``."""


def summarize(target: Dataset | Lightcone | StructureCollection) -> ResultMessage:
    """Build the result message describing ``target``.

    Parameters
    ----------
    target : Dataset, Lightcone, or StructureCollection
        The resultant object to summarize.

    Returns
    -------
    ResultMessage
        The summary message for ``target``.

    Raises
    ------
    TypeError
        If ``target`` is not a supported type.
    """
    match target:
        case Dataset():
            return DatasetSummary.from_dataset(target)
        case Lightcone():
            return LightconeSummary.from_lightcone(target)
        case StructureCollection():
            return StructureCollectionSummary.from_collection(target)
        case _:
            raise TypeError(
                f"Cannot summarize {type(target).__name__}: "
                "only Dataset, Lightcone, and StructureCollection are supported"
            )


@overload
def _summarize_child(child: Dataset) -> DatasetSummary: ...


@overload
def _summarize_child(child: Lightcone) -> LightconeSummary: ...


@overload
def _summarize_child(child: StructureCollection) -> StructureCollectionSummary: ...


def _summarize_child(
    child: Dataset | Lightcone | StructureCollection,
) -> ResultMessage:
    return summarize(child)


def _column_summaries(
    obj: Dataset | Lightcone,
) -> tuple[ColumnSummary, ...]:
    descriptions = obj.descriptions
    units = obj.units
    return tuple(
        ColumnSummary(
            name=name,
            description=descriptions.get(name),
            unit=None if (unit := units.get(name)) is None else unit.to_string(),
        )
        for name in obj.columns
    )


__all__ = [
    "ColumnSummary",
    "DatasetSummary",
    "LightconeSummary",
    "ResultMessage",
    "StructureCollectionSummary",
    "summarize",
]
