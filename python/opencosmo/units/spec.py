from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

COMOVING_FLAG_ATTR = "unit_is_comoving"
"""HDF5 column attribute indicating whether lengths in the unit are comoving.
Absent means True. Set it to False for proper (non-comoving) quantities."""


class UnitSpecDict(TypedDict):
    unit: str | None
    is_comoving: bool


@dataclass(frozen=True)
class UnitSpec:
    """Unit metadata for a single column, as stored in the file."""

    unit: str | None
    """``unit`` attribute of the column. None when the attribute is absent."""

    is_comoving: bool = True
    """Whether lengths in the unit are comoving and scale with the scale factor."""


def unit_spec_to_dict(spec: UnitSpec) -> UnitSpecDict:
    return {"unit": spec.unit, "is_comoving": spec.is_comoving}


def unit_spec_from_dict(data: UnitSpecDict) -> UnitSpec:
    return UnitSpec(unit=data["unit"], is_comoving=bool(data["is_comoving"]))
