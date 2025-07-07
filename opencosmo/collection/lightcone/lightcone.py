from typing import Optional, Self

import h5py
import numpy as np
from astropy.table import vstack  # type: ignore

import opencosmo as oc
from opencosmo.collection.protocols import Collection
from opencosmo.dataset import Dataset
from opencosmo.dataset.col import Mask
from opencosmo.io.schemas import LightconeSchema
from opencosmo.spatial import Region


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
        z_col = 1 / oc.col("fof_halo_center_a") - 1
        return dataset.with_new_columns(redshift=z_col)
    elif "redshift_true" in dataset.columns:
        z_col = 1 * oc.col("redshift_true")
        return dataset.with_new_columns(redshift=z_col)


class Lightcone(dict):
    """
    A lightcone contains two or more datasets that are part of a lightcone. Typically
    each dataset will cover a specific redshift range. The Lightcone mostly just
    delegates requested operations to the individual datasets, and provides some
    convenience methods and aliasing.
    """

    def __init__(
        self,
        datasets: dict[str, Dataset],
        z_range: Optional[tuple[float, float]] = None,
    ):
        datasets = {k: with_redshift_column(ds) for k, ds in datasets.items()}
        self.update(datasets)
        self.__z = (
            z_range
            if z_range is not None
            else get_redshift_range(list(datasets.values()))
        )

    def __len__(self):
        return sum(len(ds) for ds in self.__datasets.values())

    @property
    def dtype(self):
        return next(iter(self.values())).dtype

    @property
    def header(self):
        return next(iter(self.values())).header

    @property
    def z(self):
        return self.__z

    @property
    def data(self):
        data = [ds.data for ds in self.__datasets.values()]
        table = vstack(data, join_type="exact")
        if len(table.columns) == 1:
            return next(table.itercols())

    @property
    def cosmology(self):
        return self.__datasets[0].cosmology

    @property
    def simulation(self):
        return self.__datasets[0].header.simulation

    @classmethod
    def open(cls, handles: list[h5py.File | h5py.Group], load_kwargs):
        if len(handles) > 1:
            raise NotImplementedError
        handle = handles[0]
        datasets: dict[str, Dataset] = {}
        for key, group in handle.items():
            ds = oc.open(group)
            if not isinstance(ds, Dataset):
                raise ValueError(
                    "Lightcones can only contain datasets (not collections)"
                )
            datasets[key] = ds

        return cls(datasets)

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
        new_datasets = {}
        for key, dataset in self.items():
            if not is_in_range(dataset, z_low, z_high):
                continue
            new_dataset = dataset.filter(
                oc.col("redshift") > z_low, oc.col("redshift") < z_high
            )
            new_datasets[key] = new_dataset
        return Lightcone(new_datasets, (z_low, z_high))

    def __map(self, method, *args, **kwargs):
        """
        This type of collection will only ever be constructed if all the underlying
        datasets have the same data type, so it is always safe to map operations
        across all of them.
        """
        output = {
            k: getattr(v, method)(*args, **kwargs) for k, v in self.__datasets.items()
        }
        return Lightcone(output)

    def __map_attribute(self, attribute):
        return {k: getattr(v, attribute) for k, v in self.__datasets.items()}

    def make_schema(self) -> LightconeSchema:
        schema = LightconeSchema()
        for name, dataset in self.items():
            ds_schema = dataset.make_schema()
            schema.add_child(ds_schema, name)
        return schema

    def select(self, *args, **kwargs):
        return self.__map("select", *args, **kwargs)

    def bound(self, region: Region, select_by: Optional[str] = None):
        return self.__map("bound", region, select_by)

    def filter(self, *masks: Mask, **kwargs) -> Self:
        """
        Filter the datasets in the collection. This method behaves
        exactly like :meth:`opencosmo.Dataset.filter` or
        :meth:`opencosmo.StructureCollection.filter`, but
        it applies the filter to all the datasets or collections
        within this collection. The result is a new collection.

        Parameters
        ----------
        filters:
            The filters constructed with :func:`opencosmo.col`

        Returns
        -------
        SimulationCollection
            A new collection with the same datasets, but only the
            particles that pass the filter.
        """
        return self.__map("filter", *masks, **kwargs)

    def take(self, n: int, at: str = "random") -> Self:
        """
        Take a subest of rows from all datasets or collections in this lightcone.
        Warning: In general, lightcones are ordered by redshift slice.
        If "at" is "start" or "end", this will implictly exclude some of the
        redshift slices in this lightcone.

        Parameters
        ----------
        n: int
            The number of rows to take
        at: str, default = "random"
            The method to use to take rows. Must be one of "start", "end", "random".

        """
        raise NotImplementedError

    def with_new_columns(self, *args, **kwargs):
        """
        Update the datasets within this collection with a set of new columns.
        This method simply calls :py:meth:`opencosmo.Dataset.with_new_columns` or
        :py:meth:`opencosmo.StructureCollection.with_new_columns`, as appropriate.
        """
        return self.__map("with_new_columns", *args, **kwargs)

    def with_units(self, convention: str) -> Self:
        """
        Transform all datasets or collections to use the given unit convention. This
        method behaves exactly like :meth:`opencosmo.Dataset.with_units`.

        Parameters
        ----------
        convention: str
            The unit convention to use. One of "unitless",
            "scalefree", "comoving", or "physical".

        """
        return self.__map("with_units", convention)


test: Collection = Lightcone({})
