from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import healpy as hp
import numpy as np

from opencosmo.collection.lightcone import lightcone as lc
from opencosmo.dataset import dataset as ds

if TYPE_CHECKING:
    from astropy.table import Table


def get_redshift_range(datasets: Sequence[ds.Dataset | lc.Lightcone]):
    redshift_ranges = list(map(get_single_redshift_range, datasets))
    min_z = min(rr[0] for rr in redshift_ranges)
    max_z = max(rr[1] for rr in redshift_ranges)

    return (min_z, max_z)


def get_single_redshift_range(dataset: ds.Dataset | lc.Lightcone):
    if isinstance(dataset, lc.Lightcone):
        return dataset.z_range
    redshift_range = dataset.header.lightcone["z_range"]
    if redshift_range is not None:
        return redshift_range
    step_zs = dataset.header.simulation["step_zs"]
    step = dataset.header.file.step
    assert step is not None
    min_redshift = step_zs[step]
    max_redshift = step_zs[step - 1]
    return (min_redshift, max_redshift)


def is_in_range(dataset: ds.Dataset, z_low: float, z_high: float):
    z_range = dataset.header.lightcone["z_range"]
    if z_range is None:
        z_range = get_single_redshift_range(dataset)
    if z_high < z_range[0] or z_low > z_range[1]:
        return False
    return True


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


def determine_max_level(lightcone: lc.Lightcone) -> Optional[int]:
    """
    Return the minimum tree max_level across all datasets in the lightcone, or
    None if any dataset has no spatial index.
    """
    max_level: Optional[int] = None
    for ds_ in lightcone.values():
        if isinstance(ds_, lc.Lightcone):
            ds_level = determine_max_level(ds_)
        else:
            assert isinstance(ds_, ds.Dataset)
            ds_level = ds_.tree.max_level if ds_.tree is not None else None
        if ds_level is None:
            return None
        if max_level is None or ds_level < max_level:
            max_level = ds_level
    return max_level


def get_pixels(
    lightcone: lc.Lightcone, level: int, is_occupied: Optional[np.ndarray] = None
):
    # We know nside is a power of two at this point
    available_level = determine_max_level(lightcone)
    if available_level is None:
        raise ValueError("Lightcone does not have a spatial index!")
    if level > available_level:
        raise ValueError(
            f"The maximum available nside for this lightcone is {2**available_level}, but {2**level} was requested"
        )

    if is_occupied is None:
        is_occupied = np.zeros(hp.nside2npix(2**level), dtype=bool)

    for ds_ in lightcone.values():
        if isinstance(ds_, lc.Lightcone):
            lightcone_pixels = get_pixels(ds_, level, is_occupied)
            is_occupied[lightcone_pixels] = True
            continue

        assert isinstance(ds_, ds.Dataset)
        tree = ds_.tree
        if tree is None:
            raise ValueError(
                "One or more datasets in this lightcone does not have a spatial index!"
            )
        read_level = tree.max_level if tree.max_level < level else level
        ds_pixels = tree.get_occupied_partitions(read_level, ds_.index)
        is_occupied[ds_pixels] = True
    return np.where(is_occupied)[0]
