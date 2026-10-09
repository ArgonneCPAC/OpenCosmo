"""Rehydrate and apply validated OpenCosmo messages."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from opencosmo.collection.lightcone.healpix_map import HealpixMap
from opencosmo.collection.lightcone.lightcone import Lightcone
from opencosmo.collection.simulation.simulation import SimulationCollection
from opencosmo.collection.structure.structure import StructureCollection
from opencosmo.dataset import Dataset

from ..errors import SerdeError, make_serde_error
from .dataset import apply_dataset_message
from .expression import (
    LiveExpression,
    LiveMask,
    LiveScalar,
)
from .healpix_map import apply_healpix_map_message
from .lightcone import apply_lightcone_message
from .simulation import apply_simulation_message
from .structure import apply_structure_message

if TYPE_CHECKING:
    from ..messages import (
        DatasetMessage,
        HealpixMapMessage,
        LightconeMessage,
        SimulationMessage,
        StructureMessage,
    )


@overload
def _apply_message(target: Dataset, message: DatasetMessage) -> Dataset: ...


@overload
def _apply_message(target: Lightcone, message: LightconeMessage) -> Lightcone: ...


@overload
def _apply_message(target: HealpixMap, message: HealpixMapMessage) -> HealpixMap: ...


@overload
def _apply_message(
    target: StructureCollection, message: StructureMessage
) -> StructureCollection: ...


@overload
def _apply_message(
    target: SimulationCollection, message: SimulationMessage
) -> SimulationCollection: ...


def _apply_message(
    target: Dataset
    | HealpixMap
    | Lightcone
    | StructureCollection
    | SimulationCollection,
    message: DatasetMessage
    | HealpixMapMessage
    | LightconeMessage
    | StructureMessage
    | SimulationMessage,
) -> Dataset | HealpixMap | Lightcone | StructureCollection | SimulationCollection:
    """Apply a validated operation message to an OpenCosmo result."""
    if isinstance(target, SimulationCollection):
        return apply_simulation_message(target, message)  # type: ignore[arg-type]
    if isinstance(target, StructureCollection):
        return apply_structure_message(target, message)  # type: ignore[arg-type]
    if isinstance(target, Lightcone):
        return apply_lightcone_message(target, message)  # type: ignore[arg-type]
    if isinstance(target, HealpixMap):
        return apply_healpix_map_message(target, message)  # type: ignore[arg-type]
    if isinstance(target, Dataset):
        return apply_dataset_message(target, message)  # type: ignore[arg-type]
    raise TypeError(f"Unsupported transformation target type: {type(target).__name__}")


type ApplyMessageResponse = (
    Dataset
    | HealpixMap
    | Lightcone
    | StructureCollection
    | SimulationCollection
    | SerdeError
)


def apply_message(
    target: Dataset
    | HealpixMap
    | Lightcone
    | StructureCollection
    | SimulationCollection,
    message: DatasetMessage
    | HealpixMapMessage
    | LightconeMessage
    | StructureMessage
    | SimulationMessage,
) -> ApplyMessageResponse:
    """Apply a message, returning a structured error on failure."""
    try:
        return _apply_message(target, message)  # type: ignore[arg-type]
    except Exception as error:
        return make_serde_error(
            "apply_message",
            error,
            target_type=type(target).__name__,
            message_type=type(message).__name__,
        )


__all__ = [
    "LiveExpression",
    "LiveMask",
    "LiveScalar",
    "ApplyMessageResponse",
    "apply_message",
]
