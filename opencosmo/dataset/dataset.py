from __future__ import annotations

from typing import Generator, Optional

import h5py
import numpy as np
from astropy import units  # type: ignore
from astropy.table import Table  # type: ignore

import opencosmo.transformations as t
import opencosmo.transformations.units as u
from opencosmo.dataset.column import ColumnBuilder, get_column_builders
from opencosmo.dataset.mask import Mask, apply_masks
from opencosmo.handler import OpenCosmoDataHandler
from opencosmo.header import OpenCosmoHeader, write_header


class Dataset:
    def __init__(
        self,
        handler: OpenCosmoDataHandler,
        header: OpenCosmoHeader,
        builders: dict[str, ColumnBuilder],
        unit_transformations: dict[t.TransformationType, list[t.Transformation]],
        indices: np.ndarray,
    ):
        self.__handler = handler
        self.__header = header
        self.__builders = builders
        self.__base_unit_transformations = unit_transformations
        self.__indices = indices

    @property
    def header(self) -> OpenCosmoHeader:
        return self.__header

    @property
    def indices(self) -> np.ndarray:
        return self.__indices

    def __repr__(self):
        """
        A basic string representation of the dataset
        """
        length = len(self)
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
        return len(self.__indices)

    def __enter__(self):
        # Need to write tests
        return self

    def __exit__(self, *exc_details):
        return self.__handler.__exit__(*exc_details)

    def close(self):
        return self.__handler.__exit__()

    @property
    def cosmology(self):
        return self.__header.cosmology

    @property
    def data(self):
        # should rename this, dataset.data can get confusing
        # Also the point is that there's MORE data than just the table
        return self.__handler.get_data(builders=self.__builders, indices=self.__indices)

    def write(
        self,
        file: h5py.File,
        dataset_name: Optional[str] = None,
        with_header=True,
        _indices: Optional[np.ndarray] = None,
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
            write_header(file, self.__header, dataset_name)

        if _indices is not None:
            _indices.sort()
            if _indices[0] < 0 or _indices[-1] >= len(self):
                raise ValueError("Indices out of bounds")
            idxs = self.__indices[_indices]
        else:
            idxs = self.__indices

        self.__handler.write(file, idxs, self.__builders.keys(), dataset_name)

    def rows(self) -> Generator[dict[str, float | units.Quantity]]:
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
            chunk = self.take_range(start, end).data
            columns = {
                k: chunk[k].quantity if chunk[k].unit else chunk[k]
                for k in chunk.keys()
            }
            for i in range(len(chunk)):
                yield {k: v[i] for k, v in columns.items()}

    def take_range(self, start: int, end: int) -> Table:
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

        new_indicies = self.__indices[start:end]

        return Dataset(
            self.__handler,
            self.__header,
            self.__builders,
            self.__base_unit_transformations,
            new_indicies,
        )

    def filter(self, *masks: Mask) -> Dataset:
        """
        Filter the dataset based on some criteria.

        Parameters
        ----------
        masks : Mask
            The Masks to apply to the dataset.

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

        new_indices = apply_masks(
            self.__handler, self.__builders, masks, self.__indices
        )

        if len(new_indices) == 0:
            raise ValueError("Filter returned zero rows!")

        return Dataset(
            self.__handler,
            self.__header,
            self.__builders,
            self.__base_unit_transformations,
            new_indices,
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
            self.__header,
            new_builders,
            self.__base_unit_transformations,
            self.__indices,
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
            convention, self.__base_unit_transformations, self.__header.cosmology
        )
        new_builders = get_column_builders(new_transformations, self.__builders.keys())

        return Dataset(
            self.__handler,
            self.__header,
            new_builders,
            self.__base_unit_transformations,
            self.__indices,
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
        new_handler = self.__handler.collect(self.__builders.keys(), self.__indices)
        return Dataset(
            new_handler,
            self.__header,
            self.__builders,
            self.__base_unit_transformations,
            np.arange(len(new_handler)),
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

        if at not in ["start", "end", "random"]:
            raise ValueError(
                "Invalid value for 'at'. Must be one of 'start', 'end', or 'random'."
            )
        new_indices = self.__handler.take_indices(n, at, self.__indices)

        return Dataset(
            self.__handler,
            self.__header,
            self.__builders,
            self.__base_unit_transformations,
            new_indices,
        )
