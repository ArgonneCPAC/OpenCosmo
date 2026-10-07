"""Validated messages specific to lightcone transformations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_serializer, model_validator

from ..expression import ExpressionModel, FiniteNumber  # noqa: TC001
from .common import NonNegativeInt, PositiveInt  # noqa: TC001
from .common import validate_healpix as _validate_healpix
from .dataset import DatasetMessageType


class WithRedshiftRangeMessage(ExpressionModel):
    """A request to restrict a lightcone to a redshift interval."""

    kind: Literal["with_redshift_range"] = "with_redshift_range"
    z_low: FiniteNumber
    z_high: FiniteNumber

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        """Reject an empty redshift interval."""
        if self.z_low == self.z_high:
            raise ValueError("Low and high values of the redshift range must differ")
        return self


class PixelSearchMessage(ExpressionModel):
    """A request to restrict a lightcone to nested HEALPix pixels."""

    kind: Literal["pixel_search"] = "pixel_search"
    pixels: frozenset[NonNegativeInt] = Field(min_length=1)
    nside: PositiveInt = 64

    @model_validator(mode="after")
    def validate_healpix(self) -> Self:
        """Validate the resolution and pixel range."""
        _validate_healpix(self.pixels, self.nside)
        return self

    @field_serializer("pixels")
    def serialize_pixels(self, pixels: frozenset[int]) -> list[int]:
        """Serialize pixel identifiers in deterministic order."""
        return sorted(pixels)


type LightconeMessage = Annotated[
    DatasetMessageType | WithRedshiftRangeMessage | PixelSearchMessage,
    Field(discriminator="kind"),
]


__all__ = [
    "LightconeMessage",
    "PixelSearchMessage",
    "WithRedshiftRangeMessage",
]
