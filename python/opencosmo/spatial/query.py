from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def __coordinates(
    values: tuple[float, float, float], name: str
) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates")

    coordinates = tuple(float(value) for value in values)
    if not all(isfinite(value) for value in coordinates):
        raise ValueError(f"{name} coordinates must be finite")
    return cast("tuple[float, float, float]", coordinates)


_coordinates = __coordinates


@dataclass(frozen=True, slots=True)
class BoxQuery:
    """Normalized axis-aligned three-dimensional box query."""

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    def __post_init__(self) -> None:
        lower = _coordinates(self.lower, "lower")
        upper = _coordinates(self.upper, "upper")
        if any(
            lower_value >= upper_value for lower_value, upper_value in zip(lower, upper)
        ):
            raise ValueError(
                "box lower coordinates must be less than upper coordinates"
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class ConeQuery:
    """Normalized angular cone query with a Cartesian unit-vector center."""

    center: tuple[float, float, float]
    max_squared_chord_distance: float

    def __post_init__(self) -> None:
        center = _coordinates(self.center, "center")
        threshold = float(self.max_squared_chord_distance)
        if not isfinite(threshold) or threshold < 0:
            raise ValueError(
                "max_squared_chord_distance must be finite and nonnegative"
            )
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "max_squared_chord_distance", threshold)


@dataclass(frozen=True, slots=True)
class SkyboxQuery:
    """Normalized angular skybox query in degrees."""

    ra_start_degrees: float
    ra_width_degrees: float
    dec_min_degrees: float
    dec_max_degrees: float

    def __post_init__(self) -> None:
        values = (
            self.ra_start_degrees,
            self.ra_width_degrees,
            self.dec_min_degrees,
            self.dec_max_degrees,
        )
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("skybox coordinates must be finite")
        if not -90 <= self.dec_min_degrees <= self.dec_max_degrees <= 90:
            raise ValueError(
                "skybox declination bounds must be between -90 and 90 degrees"
            )


@dataclass(frozen=True, slots=True)
class FullSkyQuery:
    """Normalized full-sky query."""


def __nside(nside: int) -> int:
    if not isinstance(nside, int) or isinstance(nside, bool) or nside <= 0:
        raise ValueError("nside must be a positive power of two")
    if nside & (nside - 1):
        raise ValueError("nside must be a positive power of two")
    return nside


def __ranges(
    starts: NDArray[np.int64], sizes: NDArray[np.int64], nside: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    if not isinstance(starts, np.ndarray) or not isinstance(sizes, np.ndarray):
        raise ValueError("pixel ranges must be NumPy arrays")
    if starts.dtype != np.int64 or sizes.dtype != np.int64:
        raise ValueError("pixel ranges must use int64 arrays")
    if starts.ndim != 1 or sizes.ndim != 1:
        raise ValueError("pixel ranges must be one-dimensional")
    if len(starts) != len(sizes):
        raise ValueError("pixel range starts and sizes must have matching lengths")

    copied_starts = starts.copy()
    copied_sizes = sizes.copy()
    limit = 12 * nside**2
    previous_stop = 0
    for start, size in zip(copied_starts, copied_sizes):
        if start < 0 or size <= 0:
            raise ValueError(
                "pixel range starts must be nonnegative and sizes positive"
            )
        stop = int(start) + int(size)
        if stop > limit:
            raise ValueError("pixel ranges must not exceed the nside pixel count")
        if start < previous_stop:
            raise ValueError("pixel ranges must be sorted and disjoint")
        previous_stop = stop

    copied_starts.flags.writeable = False
    copied_sizes.flags.writeable = False
    return copied_starts, copied_sizes


_nside = __nside
_ranges = __ranges


@dataclass(frozen=True, slots=True)
class PixelSelection:
    """Canonical HEALPix pixel ranges with explicit resolution and ordering."""

    nside: int
    ordering: Literal["nested", "ring"]
    starts: NDArray[np.int64]
    sizes: NDArray[np.int64]

    def __post_init__(self) -> None:
        nside = _nside(self.nside)
        if self.ordering not in ("nested", "ring"):
            raise ValueError("pixel ordering must be 'nested' or 'ring'")
        starts, sizes = _ranges(self.starts, self.sizes, nside)
        object.__setattr__(self, "nside", nside)
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "sizes", sizes)


def validate_nested_promotion(source_nside: int, target_nside: int) -> int:
    """Validate nested HEALPix promotion and return its integer resolution factor."""
    source_nside = __nside(source_nside)
    target_nside = __nside(target_nside)
    if target_nside < source_nside or target_nside % source_nside:
        raise ValueError("target nside must be a compatible nested resolution")
    return target_nside // source_nside


NormalizedRegion: TypeAlias = (
    BoxQuery | ConeQuery | SkyboxQuery | FullSkyQuery | PixelSelection
)
