"""Rehydrate serialized spatial regions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opencosmo.spatial.builders import make_box, make_cone, make_skybox
from opencosmo.spatial.region import HealpixRegion

from ..messages import (
    BoxRegionMessage,
    ConeRegionMessage,
    HealpixRegionMessage,
    SkyboxRegionMessage,
)

if TYPE_CHECKING:
    from opencosmo.spatial.protocols import Region

    from ..messages import RegionMessage


def region_to_live(region: RegionMessage) -> Region:
    """Build an OpenCosmo region from a serialized region message."""
    match region:
        case BoxRegionMessage(p1=p1, p2=p2):
            return make_box(p1, p2)
        case ConeRegionMessage(center=center, radius=radius):
            return make_cone(center, radius)
        case SkyboxRegionMessage(p1=p1, p2=p2):
            return make_skybox(p1, p2)
        case HealpixRegionMessage(pixels=pixels, nside=nside):
            return HealpixRegion(np.asarray(sorted(pixels), dtype=np.int64), nside)


__all__ = ["region_to_live"]
