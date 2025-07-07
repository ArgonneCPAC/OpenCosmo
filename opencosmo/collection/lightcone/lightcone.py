from typing import Optional

import numpy as np

import opencosmo as oc
from opencosmo.dataset import Dataset
from opencosmo.header import OpenCosmoHeader


def get_redshift_range(datasets: list[Dataset]):
    steps = np.fromiter((ds.header.file.step for ds in datasets), dtype=int)
    step_zs = datasets[0].header.simulation.step_zs
    min_step = np.min(steps)
    max_step = np.max(steps)

    min_redshift = step_zs[max_step]
    max_redshift = step_zs[min_step - 1]
    return (min_redshift, max_redshift)


def is_in_range(dataset: Dataset, z_low: float, z_high: float):
    step_zs = dataset.header.simulation.step_zs
    z_range = (step_zs[dataset.header.file.step], step_zs[dataset.header.file.step - 1])
    if z_high < z_range[0] or z_low > z_range[1]:
        return False
    return True


def with_redshift_column(dataset: Dataset):
    """
    Ensures a column exists called "redshift" which contains the redshift of the objects
    in the lightcone.
    """
    if "redshift" in dataset.columns:
        return dataset

    elif "fof_halo_center_a" in dataset.columns:
        z_col = 1 - 1 / oc.col("fof_halo_center_a")
        return dataset.with_new_columns(redshift=z_col)
    elif "redshift_true" in dataset.columns:
        z_col = 1 * oc.col("redshift_true")
        return dataset.with_new_columns(redshift=z_col)


class Lightcone:
    """
    A lightcone contains two or more datasets that are part of a lightcone. Typically
    each dataset will cover a specific redshift range. The Lightcone mostly just
    delegates requested operations to the individual datasets, and provides some
    convenience methods and aliasing.
    """

    def __init__(
        self,
        header: OpenCosmoHeader,
        datasets: list[Dataset],
        z_range: Optional[tuple[float, float]] = None,
    ):
        self.__datasets = [with_redshift_column(ds) for ds in datasets]
        self.__z = z_range if z_range is not None else get_redshift_range(datasets)

    @property
    def cosmology(self):
        return self.__datasets[0].cosmology

    @property
    def simulation(self):
        return self.__datasets[0].header.simulation

    def with_redshift_range(self, z_low: float, z_high: float):
        """
        Restrict this lightcone to a specific redshift range.
        """
        if z_high < z_low:
            z_high, z_low = z_low, z_high

        if z_high < self.__z[0] or z_low > self.__z[1]:
            raise ValueError(
                "This lightcone only ranges from "
                f"z = {self.__z[0]} to z = {self.__z[1]}"
            )

        elif z_low == z_high:
            raise ValueError("Low and high values of the redshift range are the same!")
