from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from opencosmo.spatial.query import (
    BoxQuery,
    FullSkyQuery,
    PixelSelection,
)

if TYPE_CHECKING:
    from opencosmo.spatial.query import NormalizedRegion


@dataclass(frozen=True, slots=True)
class SpatialNormalizationContext:
    """Dataset-specific metadata needed to normalize a spatial query."""

    dimensions: Literal[2, 3]
    coordinate_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.dimensions not in (2, 3):
            raise ValueError("spatial dimensions must be 2 or 3")
        coordinate_names = tuple(self.coordinate_names)
        if len(coordinate_names) != self.dimensions:
            raise ValueError("coordinate name count must match spatial dimensions")
        if not all(isinstance(name, str) and name for name in coordinate_names):
            raise ValueError("coordinate names must be nonempty strings")
        object.__setattr__(self, "coordinate_names", coordinate_names)


def normalize_region(
    region: object, *, context: SpatialNormalizationContext
) -> NormalizedRegion:
    """Convert supported public regions into immutable numeric query specifications."""
    from opencosmo.spatial.region import BoxRegion, FullSkyRegion, HealpixRegion

    match region:
        case BoxRegion():
            if context.dimensions != 3:
                raise ValueError("box regions require a three-dimensional context")
            lower, upper = zip(*region.bounds)
            return BoxQuery(lower, upper)
        case HealpixRegion():
            if context.dimensions != 2:
                raise ValueError("HEALPix regions require a two-dimensional context")
            pixels = region.pixels
            if pixels.size == 0:
                starts = np.array([], dtype=np.int64)
                sizes = np.array([], dtype=np.int64)
            else:
                starts = pixels[np.r_[True, np.diff(pixels) != 1]].astype(np.int64)
                stops = pixels[np.r_[np.diff(pixels) != 1, True]].astype(np.int64) + 1
                sizes = stops - starts
            return PixelSelection(region.nside, region.ordering, starts, sizes)
        case FullSkyRegion():
            if context.dimensions != 2:
                raise ValueError("full-sky regions require a two-dimensional context")
            return FullSkyQuery()
        case _:
            raise TypeError(f"unsupported region type {type(region).__name__}")
