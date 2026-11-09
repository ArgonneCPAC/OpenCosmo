"""Shared wire types and validation for messages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import astropy.units as u
import healpy as hp
from pydantic import AfterValidator, Field, StrictInt

if TYPE_CHECKING:
    from collections.abc import Collection


def normalize_unit(value: str) -> str:
    """Validate and normalize an Astropy unit string."""
    if not value:
        raise ValueError("Unit strings must not be empty")
    try:
        return u.Unit(value).to_string()
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid Astropy unit: {value!r}") from error


def validate_healpix(pixels: Collection[int], nside: int) -> None:
    """Validate nested HEALPix resolution and pixel identifiers."""
    if not hp.isnsideok(nside) or nside & (nside - 1):
        raise ValueError("nside must be a positive power of two")
    pixel_count = hp.nside2npix(nside)
    if any(pixel >= pixel_count for pixel in pixels):
        raise ValueError(f"pixels must be less than {pixel_count} for this nside")


type NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
type PositiveInt = Annotated[StrictInt, Field(gt=0)]
type Keyword = Annotated[str, Field(min_length=1)]
type UnitString = Annotated[str, AfterValidator(normalize_unit)]
type RowIndices = tuple[NonNegativeInt, ...]


__all__ = [
    "Keyword",
    "NonNegativeInt",
    "PositiveInt",
    "RowIndices",
    "UnitString",
    "normalize_unit",
    "validate_healpix",
]
