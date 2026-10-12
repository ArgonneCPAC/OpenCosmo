from __future__ import annotations

from typing import TYPE_CHECKING

import healpy as hp
import numpy as np

from opencosmo.spatial.index import get_partitions_with_data

if TYPE_CHECKING:
    from opencosmo.collection.lightcone.state import LightconeState


def get_required_columns(columns: set[str], dtype: str) -> set[str]:
    """
    Columns that must be retained (even if hidden) for a lightcone to be written
    correctly, given the columns available in the underlying datasets.

    "redshift" is needed because lightcones are stacked by redshift, and the
    angular coordinates (ra/dec, or theta/phi) are needed because stacking
    pixelizes rows by their sky position at write time.

    Particles and profiles are excluded: their redshift and coordinates are
    handled at the structure-collection level, mirroring the open-time hooks.
    """
    if "particles" in dtype or "profiles" in dtype:
        return set()

    required: set[str] = set()

    if "redshift" in columns:
        required.add("redshift")

    # Angular coordinates are required for stacking. Prefer ra/dec, but fall back
    # to theta/phi if that is all the dataset carries.
    if {"ra", "dec"}.issubset(columns):
        required.update({"ra", "dec"})
    elif {"theta", "phi"}.issubset(columns):
        required.update({"theta", "phi"})

    return required


def determine_max_level(state: LightconeState) -> int | None:
    """
    Return the minimum tree max_level across all datasets in the lightcone, or
    None if any dataset has no spatial index.
    """
    levels = [
        leaf.spatial_index.level if leaf.spatial_index is not None else None
        for _, leaf in state.leaves
    ]
    if any(level is None for level in levels):
        return None
    return min(levels)  # type: ignore[type-var]


def get_pixels(state: LightconeState, level: int) -> np.ndarray:
    # We know nside is a power of two at this point
    available_level = determine_max_level(state)
    if available_level is None:
        raise ValueError("Lightcone does not have a spatial index!")
    if level > available_level:
        raise ValueError(
            f"The maximum available nside for this lightcone is {2**available_level}, but {2**level} was requested"
        )

    is_occupied = np.zeros(hp.nside2npix(2**level), dtype=bool)
    for _, leaf in state.leaves:
        assert leaf.spatial_index is not None
        read_level = min(leaf.spatial_index.level, level)
        leaf_pixels = get_partitions_with_data(
            leaf.spatial_index, read_level, leaf.raw_index
        )
        is_occupied[leaf_pixels] = True
    return np.where(is_occupied)[0]
