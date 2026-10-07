"""Common types and validation for transformation messages."""

from enum import StrEnum
from typing import Annotated

import astropy.units as u
import healpy as hp
from pydantic import Field, StrictInt

type ColumnName = Annotated[str, Field(min_length=1)]
type NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
type PositiveInt = Annotated[StrictInt, Field(gt=0)]


class ReductionMode(StrEnum):
    """Scope used to compute scalar reductions."""

    LOCAL = "local"
    GLOBAL = "global"


class TakePosition(StrEnum):
    """Position from which rows are selected."""

    START = "start"
    END = "end"
    RANDOM = "random"


def normalize_unit(value: str) -> str:
    """Validate and normalize an Astropy unit string."""
    if not value:
        raise ValueError("Unit strings must not be empty")
    try:
        return u.Unit(value).to_string()
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid Astropy unit: {value!r}") from error


def validate_healpix(pixels: frozenset[int], nside: int) -> None:
    """Validate nested HEALPix resolution and pixel identifiers."""
    if not hp.isnsideok(nside) or nside & (nside - 1):
        raise ValueError("nside must be a positive power of two")
    pixel_count = hp.nside2npix(nside)
    if any(pixel >= pixel_count for pixel in pixels):
        raise ValueError(f"pixels must be less than {pixel_count} for this nside")


__all__ = [
    "ColumnName",
    "NonNegativeInt",
    "PositiveInt",
    "ReductionMode",
    "TakePosition",
]
