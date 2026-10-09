"""Validated spatial-region messages."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_serializer, model_validator

from ..expression import ExpressionModel, FiniteNumber  # noqa: TC001
from .common import NonNegativeInt, PositiveInt, validate_healpix  # noqa: TC001


class BoxRegionMessage(ExpressionModel):
    """A three-dimensional box query."""

    kind: Literal["box"] = "box"
    p1: tuple[FiniteNumber, FiniteNumber, FiniteNumber]
    p2: tuple[FiniteNumber, FiniteNumber, FiniteNumber]


class ConeRegionMessage(ExpressionModel):
    """A circular sky query in degrees."""

    kind: Literal["cone"] = "cone"
    center: tuple[FiniteNumber, FiniteNumber]
    radius: Annotated[FiniteNumber, Field(gt=0)]


class SkyboxRegionMessage(ExpressionModel):
    """A rectangular sky query in degrees."""

    kind: Literal["skybox"] = "skybox"
    p1: tuple[FiniteNumber, FiniteNumber]
    p2: tuple[FiniteNumber, FiniteNumber]


class HealpixRegionMessage(ExpressionModel):
    """A HEALPix query containing explicit nested pixel identifiers."""

    kind: Literal["healpix"] = "healpix"
    pixels: frozenset[NonNegativeInt]
    nside: PositiveInt

    @model_validator(mode="after")
    def validate_healpix(self) -> Self:
        """Validate the resolution and pixel range."""
        validate_healpix(self.pixels, self.nside)
        return self

    @field_serializer("pixels")
    def serialize_pixels(self, pixels: frozenset[int]) -> list[int]:
        """Serialize pixel identifiers in deterministic order."""
        return sorted(pixels)


type RegionMessage = Annotated[
    BoxRegionMessage | ConeRegionMessage | SkyboxRegionMessage | HealpixRegionMessage,
    Field(discriminator="kind"),
]


__all__ = [
    "BoxRegionMessage",
    "ConeRegionMessage",
    "HealpixRegionMessage",
    "RegionMessage",
    "SkyboxRegionMessage",
]
