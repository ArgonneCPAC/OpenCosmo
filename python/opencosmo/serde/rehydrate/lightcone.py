"""Apply serialized transformation messages to lightcones."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..messages import (
    BoundMessage,
    BoxRegionMessage,
    HealpixRegionMessage,
    PixelSearchMessage,
    WithRedshiftRangeMessage,
)
from .dataset import apply_dataset_message

if TYPE_CHECKING:
    from opencosmo.collection.lightcone.lightcone import Lightcone

    from ..messages import DatasetMessage, LightconeMessage


def apply_lightcone_message(
    lightcone: Lightcone,
    message: LightconeMessage | DatasetMessage,
) -> Lightcone:
    """Apply a validated transformation message to a lightcone."""
    match message:
        case WithRedshiftRangeMessage(z_low=z_low, z_high=z_high):
            return lightcone.with_redshift_range(z_low, z_high)
        case PixelSearchMessage(pixels=pixels, nside=nside):
            return lightcone.pixel_search(
                np.asarray(sorted(pixels), dtype=np.int64), nside=nside
            )
        case BoundMessage(region=BoxRegionMessage()):
            raise ValueError(
                "Three-dimensional box bounds are not supported by Lightcone"
            )
        case BoundMessage(region=HealpixRegionMessage()) if lightcone.map is not None:
            raise ValueError(
                "HEALPix region bounds are not supported by Lightcones with attached maps; "
                "use PixelSearchMessage for catalog-only pixel pruning"
            )
        case _:
            result = apply_dataset_message(lightcone, message)  # type: ignore[arg-type]
            if not isinstance(result, type(lightcone)):
                raise RuntimeError("Lightcone operation produced an invalid result")
            return result


__all__ = ["apply_lightcone_message"]
