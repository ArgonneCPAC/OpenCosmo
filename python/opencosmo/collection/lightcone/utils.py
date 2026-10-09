from __future__ import annotations

from typing import TYPE_CHECKING

import healpy as hp
import numpy as np

from opencosmo.collection.lightcone import lightcone as lc
from opencosmo.collection.lightcone.state import header_z_range
from opencosmo.spatial.index import get_partitions_with_data

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astropy.table import Table

    from opencosmo.collection.lightcone.state import LightconeState
    from opencosmo.dataset import dataset as ds


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


def get_redshift_range(datasets: Sequence[ds.Dataset | lc.Lightcone]):
    redshift_ranges = list(map(get_single_redshift_range, datasets))
    min_z = min(rr[0] for rr in redshift_ranges)
    max_z = max(rr[1] for rr in redshift_ranges)

    return (min_z, max_z)


def get_single_redshift_range(dataset: ds.Dataset | lc.Lightcone):
    if isinstance(dataset, lc.Lightcone):
        return dataset.z_range
    return header_z_range(dataset.header)


def sort_table(table: Table, column: str, invert: bool):
    column_data = table[column]
    if invert:
        column_data = -column_data
    indices = np.argsort(column_data)
    for name in table.columns:
        table[name] = table[name][indices]
    return table


def take_from_sorted(
    lightcone: lc.Lightcone, sort_by: str, invert: bool, n: int, at: str | int
):
    column = np.concatenate(
        [ds.select(sort_by).get_data("numpy") for ds in lightcone.values()]
    )
    if invert:
        column = -column
    sort_index = np.argsort(column)
    if at == "start":
        sort_index = sort_index[:n]
    elif at == "end":
        sort_index = sort_index[-n:]
    elif isinstance(at, int):
        if at + n > len(sort_index) or at < 0:
            raise ValueError(
                "Requested a range that is outside the size of this dataset!"
            )
        sort_index = sort_index[at : at + n]

    sorted_indices = np.sort(sort_index)
    return sorted_indices


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
