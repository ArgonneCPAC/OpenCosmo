from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from opencosmo.collection.lightcone import lightcone as oclc

if TYPE_CHECKING:
    from astropy.table import Table

    from opencosmo import Dataset


def get_redshift_range(datasets: Sequence[Dataset | oclc.Lightcone]):
    redshift_ranges = list(map(get_single_redshift_range, datasets))
    min_z = min(rr[0] for rr in redshift_ranges)
    max_z = max(rr[1] for rr in redshift_ranges)

    return (min_z, max_z)


def get_single_redshift_range(dataset: Dataset | oclc.Lightcone):
    if isinstance(dataset, oclc.Lightcone):
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


def is_in_range(dataset: Dataset, z_low: float, z_high: float):
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
    lightcone: "oclc.Lightcone", sort_by: str, invert: bool, n: int, at: str | int
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
