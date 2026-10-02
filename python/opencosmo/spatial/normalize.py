from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from opencosmo.spatial.query import (
    BoxQuery,
    ConeQuery,
    FullSkyQuery,
    PixelSelection,
    SkyboxQuery,
)

if TYPE_CHECKING:
    from opencosmo.spatial.query import NormalizedRegion
    from opencosmo.units.handler import UnitHandler


@dataclass(frozen=True, slots=True)
class SpatialNormalizationContext:
    """Dataset-specific metadata needed to normalize a spatial query."""

    dimensions: Literal[2, 3]
    coordinate_names: tuple[str, ...]
    unit_handler: UnitHandler | None = None
    unit_kwargs: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.dimensions not in (2, 3):
            raise ValueError("spatial dimensions must be 2 or 3")
        coordinate_names = tuple(self.coordinate_names)
        if len(coordinate_names) != self.dimensions:
            raise ValueError("coordinate name count must match spatial dimensions")
        if not all(isinstance(name, str) and name for name in coordinate_names):
            raise ValueError("coordinate names must be nonempty strings")
        unit_kwargs = tuple(
            (str(name), float(value)) for name, value in self.unit_kwargs
        )
        if not all(np.isfinite(value) for _, value in unit_kwargs):
            raise ValueError("unit conversion values must be finite")
        object.__setattr__(self, "coordinate_names", coordinate_names)
        object.__setattr__(self, "unit_kwargs", unit_kwargs)


def normalize_region(
    region: object, *, context: SpatialNormalizationContext
) -> NormalizedRegion:
    """Convert supported public regions into immutable numeric query specifications."""
    from opencosmo.spatial.region import (
        BoxRegion,
        ConeRegion,
        FullSkyRegion,
        HealpixRegion,
        SkyboxRegion,
    )

    match region:
        case BoxRegion():
            if context.dimensions != 3:
                raise ValueError("box regions require a three-dimensional context")
            lower, upper = zip(*region.bounds)
            if context.unit_handler is not None:
                lower = tuple(
                    value.value
                    for value in context.unit_handler.into_base_convention(
                        dict(zip(context.coordinate_names, lower)),
                        dict(context.unit_kwargs),
                    ).values()
                )
                upper = tuple(
                    value.value
                    for value in context.unit_handler.into_base_convention(
                        dict(zip(context.coordinate_names, upper)),
                        dict(context.unit_kwargs),
                    ).values()
                )
            return BoxQuery(lower, upper)
        case HealpixRegion():
            if context.dimensions != 2:
                raise ValueError("HEALPix regions require a two-dimensional context")
            pixels = np.unique(region.pixels)
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
        case ConeRegion():
            if context.dimensions != 2:
                raise ValueError("cone regions require a two-dimensional context")
            center = region.center.cartesian.xyz.value
            radius = region.radius.to_value("rad")
            return ConeQuery(
                (float(center[0]), float(center[1]), float(center[2])),
                float(2 - 2 * np.cos(radius)),
            )
        case SkyboxRegion():
            if context.dimensions != 2:
                raise ValueError("skybox regions require a two-dimensional context")
            if region.ra_width == 0:
                raise ValueError("skybox RA interval must have nonzero width")
            interval = cast(
                "Literal['wrapped', 'non_wrapped']",
                "wrapped" if region.ra_start + region.ra_width > 360 else "non_wrapped",
            )
            return SkyboxQuery(
                region.ra_start,
                region.ra_width,
                *region.dec_bounds,
                interval,
            )
        case _:
            raise TypeError(f"unsupported region type {type(region).__name__}")
