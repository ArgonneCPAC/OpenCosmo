from __future__ import annotations

from copy import copy
from typing import Optional

import astropy.units as u
import h5py
import numpy as np
from astropy.table import Table

import opencosmo.transformations as t
from opencosmo.dataset.column import ColumnBuilder, get_column_builders
from opencosmo.dataset.mask import Mask, apply_masks
from opencosmo.handler import OpenCosmoDataHandler
from opencosmo.header import OpenCosmoHeader, write_header
from opencosmo.transformations import units as u


class Dataset:
    def __init__(
        self,
        handler: OpenCosmoDataHandler,
        header: OpenCosmoHeader,
        builders: dict[str, ColumnBuilder],
        unit_transformations: dict[t.TransformationType, list[t.Transformation]],
        mask: np.ndarray,
    ):
        self.__handler = handler
        self.__header = header
        self.__builders = builders
        self.__base_unit_transformations = unit_transformations
        self.__mask = mask

    @property
    def header(self) -> OpenCosmoHeader:
        return self.__header

    @property
    def mask(self) -> np.ndarray:
        return self.__mask

    def __repr__(self):
        """
        A basic string representation of the dataset
        """
        length = np.sum(self.__mask)
        take_length = length if length < 10 else 10
        repr_ds = self.take(take_length)
        table_repr = repr_ds.data.__repr__()
        # remove the first line
        table_repr = table_repr[table_repr.find("\n") + 1 :]
        head = f"OpenCosmo Dataset (length={length})\n"
        cosmo_repr = f"Cosmology: {self.cosmology.__repr__()}" + "\n"
        table_head = f"First {take_length} rows:\n"
        return head + cosmo_repr + table_head + table_repr

    def __len__(self):
        return np.sum(self.__mask)

    def __enter__(self):
        # Need to write tests
        return self

    def __exit__(self, *exc_details):
        return self.__handler.__exit__(*exc_details)

    def close(self):
        return self.__handler.__exit__()

    @property
    def cosmology(self):
        return self.header.cosmology

    @property
    def data(self):
        # should rename this, dataset.data can get confusing
        # Also the point is that there's MORE data than just the table
        return self.__handler.get_data(builders=self.__builders, mask=self.__mask)

    def write(
        self, file: h5py.File, dataset_name: Optional[str] = None, with_header=True
    ) -> None:
        """
        Write the dataset to a file. This should not be called directly for the user.
        The opencosmo.write file writer automatically handles the file context.

        Parameters
        ----------
        file : h5py.File
            The file to write to.
        dataset_name : str
            The name of the dataset in the file. The default is "data".

        """
        if not isinstance(file, h5py.File):
            raise AttributeError(
                "Dataset.write should not be called directly, "
                "use opencosmo.write instead."
            )

        if with_header:
            write_header(file, self.header, dataset_name)

        self.__handler.write(file, self.__mask, self.__builders.keys(), dataset_name)

    def rows(self) -> dict[str, float | u.Quantity]:
        """
        Iterate over the rows in the dataset. Returns a dictionary of values
        for each row, with associated units. For performance it is recommended
        that you first select the columns you need to work with.

        Yields
        -------
        row : dict
            A dictionary of values for each row in the dataset.
        """
        max = len(self)
        chunk_ranges = [(i, min(i + 1000, max)) for i in range(0, max, 1000)]
        for start, end in chunk_ranges:
            chunk = self.get_range(start, end)
            columns = {
                k: chunk[k].quantity if chunk[k].unit else chunk[k]
                for k in chunk.keys()
            }
            for i in range(len(chunk)):
                yield {k: v[i] for k, v in columns.items()}

    def get_range(self, start: int, end: int) -> Table:
        """
        Get a range of rows from the dataset.

        Parameters
        ----------
        start : int
            The first row to get.
        end : int
            The last row to get.

        Returns
        -------
        table : astropy.table.Table
            The table with only the rows from start to end.

        Raises
        ------
        ValueError
            If start or end are negative, or if end is greater than start.

        """
        if start < 0 or end < 0:
            raise ValueError("start and end must be positive.")
        if end < start:
            raise ValueError("end must be greater than start.")
        if end > len(self):
            raise ValueError("end must be less than the length of the dataset.")

        return self.__handler.get_range(start, end, self.__builders, self.__mask)

    def filter(self, *masks: Mask, boolean_filter: np.ndarray = None) -> Dataset:
        """
        Filter the dataset based on some criteria.

        Parameters
        ----------
        s : mask
            The s to apply to the dataset.

        Returns
        -------
        dataset : Dataset
            The new dataset with the s applied.

        Raises
        ------
        ValueError
            If the given  refers to columns that are
            not in the dataset, or the  would return zero rows.

        """
        if boolean_filter is not None:
            if len(boolean_filter) != sum(self.__mask):
                raise ValueError(
                    "Boolean filter must be the same length as the dataset."
                )
            elif boolean_filter.dtype != bool:
                raise ValueError("Boolean filter must be a boolean array.")
            new_mask = copy(self.__mask)
            new_mask[self.__mask] = boolean_filter

        else:
            new_mask = apply_masks(self.__handler, self.__builders, masks, self.__mask)
            if np.sum(new_mask) == 0:
                raise ValueError(" would return zero rows.")

        return Dataset(
            self.__handler,
            self.header,
            self.__builders,
            self.__base_unit_transformations,
            new_mask,
        )

    def select(self, columns: str | list[str]) -> Dataset:
        """
        Select a subset of columns from the dataset.

        Parameters
        ----------
        columns : str or list of str
            The column or columns to select.

        Returns
        -------
        dataset : Dataset
            The new dataset with only the selected columns.

        Raises
        ------
        ValueError
            If any of the given columns are not in the dataset.
        """
        if isinstance(columns, str):
            columns = [columns]

        # numpy compatability
        columns = [str(col) for col in columns]

        try:
            new_builders = {col: self.__builders[col] for col in columns}
        except KeyError:
            known_columns = set(self.__builders.keys())
            unknown_columns = set(columns) - known_columns
            raise ValueError(
                "Tried to select columns that aren't in this dataset! Missing columns "
                + ", ".join(unknown_columns)
            )

        return Dataset(
            self.__handler,
            self.header,
            new_builders,
            self.__base_unit_transformations,
            self.__mask,
        )

    def with_units(self, convention: str) -> Dataset:
        """
        Transform this dataset to a different unit convention

        Parameters
        ----------
        convention : str
            The unit convention to use. One of "physical", "comoving",
            "scalefree", or "unitless".

        Returns
        -------
        dataset : Dataset
            The new dataset with the requested unit convention.

        """
        new_transformations = u.get_unit_transition_transformations(
            convention, self.__base_unit_transformations, self.header.cosmology
        )
        new_builders = get_column_builders(new_transformations, self.__builders.keys())

        return Dataset(
            self.__handler,
            self.header,
            new_builders,
            self.__base_unit_transformations,
            self.__mask,
        )

    def collect(self) -> Dataset:
        """
        Given a dataset that was originally opend with opencosmo.open,
        return a dataset that is in-memory as though it was read with
        opencosmo.read.

        This is useful if you have a very large dataset on disk, and you
        want to filter it down and then close the file.

        For example:

        .. code-block:: python

            import opencosmo as oc
            with oc.open("path/to/file.hdf5") as file:
                ds = file.(ds["sod_halo_mass"] > 0)
                ds = ds.select(["sod_halo_mass", "sod_halo_radius"])
                ds = ds.collect()

        The selected data will now be in memory, and the file will be closed.

        If working in an MPI context, all ranks will recieve the same data.
        """
        new_handler = self.__handler.collect(self.__builders.keys(), self.__mask)
        return Dataset(
            new_handler,
            self.header,
            self.__builders,
            self.__base_unit_transformations,
            np.ones(len(new_handler), dtype=bool),
        )

    def take(self, n: int, at: str = "start") -> Dataset:
        """
        Take n rows from the dataset.

        Can take the first n rows, the last n rows, or n random rows
        depending on the value of 'at'.

        Parameters
        ----------
        n : int
            The number of rows to take.
        at : str
            Where to take the rows from. One of "start", "end", or "random".
            The default is "start".

        Returns
        -------
        dataset : Dataset
            The new dataset with only the first n rows.

        Raises
        ------
        ValueError
            If n is negative or greater than the number of rows in the dataset,
            or if 'at' is invalid.

        """
        new_mask = self.__handler.take_mask(n, at, self.__mask)
        if np.sum(new_mask) == 0:
            # This should only happen in an MPI context, so
            # delegate error handling to the user.
            raise ValueError("Filter returned zero rows!")

        return Dataset(
            self.__handler,
            self.header,
            self.__builders,
            self.__base_unit_transformations,
            new_mask,
        )
