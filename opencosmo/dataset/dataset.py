from __future__ import annotations

import h5py
import numpy as np

import opencosmo.transformations as t
from opencosmo.dataset.column import ColumnBuilder, get_column_builders
from opencosmo.dataset.filter import Filter, apply_filters
from opencosmo.file import file_reader
from opencosmo.handler import InMemoryHandler, OpenCosmoDataHandler
from opencosmo.header import OpenCosmoHeader, read_header
from opencosmo.transformations import units as u


@file_reader
def read(file: h5py.File, units: str = "comoving") -> Dataset:
    """
    Read a dataset from a file into memory.

    You should use this function if the data are small enough that having
    a copy of it (or a few copies of it) in memory is not a problem. For
    larger datasets, use opencosmo.open.

    Parameters
    ----------
    file : str or pathlib.Path
        The path to the file to read.
    units : str | None
        The unit convention to use. One of "physical", "comoving",
        "scalefree", or None. The default is "comoving".

    Returns
    -------
    dataset : Dataset
        The dataset read from the file.

    """
    header = read_header(file)
    handler = InMemoryHandler(file)
    base_unit_transformations, transformations = u.get_unit_transformations(
        file["data"], header, units
    )
    column_names = list(str(col) for col in file["data"].keys())
    builders = get_column_builders(transformations, column_names)
    filter = np.ones(len(handler), dtype=bool)

    return Dataset(handler, header, builders, base_unit_transformations, filter)


class Dataset:
    def __init__(
        self,
        handler: OpenCosmoDataHandler,
        header: OpenCosmoHeader,
        builders: dict[str, ColumnBuilder],
        unit_transformations: dict[t.TransformationType, list[t.Transformation]],
        filter: np.ndarray,
    ):
        self.__header = header
        self.__handler = handler
        self.__builders = builders
        self.__base_unit_transformations = unit_transformations
        self.__filter = filter

    def __repr__(self):
        length = np.sum(self.__filter)
        take_length = length if length < 10 else 10
        repr_ds = self.take(take_length)
        table_repr = repr_ds.data.__repr__()
        # remove the first line
        table_repr = table_repr[table_repr.find("\n") + 1 :]
        head = f"OpenCosmo Dataset (length={length})\n"
        cosmo_repr = f"Cosmology: {self.cosmology.__repr__()}" + "\n"
        table_head = f"First {take_length} rows:\n"
        return head + cosmo_repr + table_head + table_repr

    def __enter__(self):
        # Need to write tests
        return self

    def __exit__(self, *exc_details):
        return self.__handler.__exit__(*exc_details)

    @property
    def cosmology(self):
        return self.__header.cosmology

    @property
    def data(self):
        # should rename this, dataset.data can get confusing
        # Also the point is that there's MORE data than just the table
        return self.__handler.get_data(builders=self.__builders, filter=self.__filter)

    def filter(self, *filters: Filter) -> Dataset:
        """
        Filter the dataset based on some criteria.

        Parameters
        ----------
        filters : Filter
            The filters to apply to the dataset.

        Returns
        -------
        dataset : Dataset
            The new dataset with the filters applied.

        Raises
        ------
        ValueError
            If the given filter refers to columns that are
            not in the dataset, or the filter would return zero rows.

        """
        new_filter = apply_filters(
            self.__handler, self.__builders, filters, self.__filter
        )
        if np.sum(new_filter) == 0:
            raise ValueError("Filter would return zero rows.")

        return Dataset(
            self.__handler,
            self.__header,
            self.__builders,
            self.__base_unit_transformations,
            new_filter,
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

        if not all(col in self.__builders for col in columns):
            raise ValueError("Not all columns are present in the dataset.")

        new_builders = {col: self.__builders[col] for col in columns}

        return Dataset(
            self.__handler,
            self.__header,
            new_builders,
            self.__base_unit_transformations,
            self.__filter,
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
            self.__filter,
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
        if n < 0 or n > np.sum(self.__filter):
            raise ValueError("Invalid number of rows to take.")

        new_filter = np.zeros_like(self.__filter)
        indices = np.where(self.__filter)[0]
        if at == "start":
            new_filter[indices[:n]] = True
        elif at == "end":
            new_filter[indices[-n:]] = True
        elif at == "random":
            indices = np.random.choice(indices, n, replace=False)
            new_filter[indices] = True
        else:
            raise ValueError(
                "Invalid value for 'at'. Expected 'start', 'end', or 'random'."
            )

        return Dataset(
            self.__handler,
            self.__header,
            self.__builders,
            self.__base_unit_transformations,
            new_filter,
        )
