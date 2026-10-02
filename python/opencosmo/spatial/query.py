from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias, cast


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


NormalizedRegion: TypeAlias = BoxQuery | ConeQuery | SkyboxQuery | FullSkyQuery
