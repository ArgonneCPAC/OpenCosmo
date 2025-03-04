from __future__ import annotations

import h5py

from opencosmo.file import file_reader
from opencosmo.handler import InMemoryHandler, OpenCosmoDataHandler
from opencosmo.header import OpenCosmoHeader, read_header
from opencosmo.transformations import TransformationType
from opencosmo.transformations import units as u
from opencosmo.transformations.select import select_columns


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

    return Dataset(handler, header, base_unit_transformations, transformations)


class Dataset:
    def __init__(
        self,
        handler: OpenCosmoDataHandler,
        header: OpenCosmoHeader,
        unit_transformations: dict = {},
        transformations: dict = {},
    ):
        self.__header = header
        self.__handler = handler
        self.__unit_transformations = unit_transformations
        self.__transformations = transformations

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
        return self.__handler.get_data(transformations=self.__transformations)

    def select(self, columns: str | list[str]) -> Dataset:
        """
        Select a subset of columns from the dataset.

        Parameters
        ----------
        columns : str or list of str
            The columns to select.

        Returns
        -------
        dataset : Dataset
            The new dataset with only the selected columns.

        """
        if isinstance(columns, str):
            columns = [columns]

        new_transformation = select_columns(columns)
        new_transformations = self.__transformations.copy()
        current_table_transformations = new_transformations.get(
            TransformationType.TABLE, []
        )
        new_table_transformations = current_table_transformations + [new_transformation]
        new_transformations[TransformationType.TABLE] = new_table_transformations
        return Dataset(
            self.__handler,
            self.__header,
            self.__unit_transformations,
            new_transformations,
        )

    def with_units(self, convention: str) -> Dataset:
        """
        Get a new dataset with a different unit convention.

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
            convention, self.__unit_transformations, self.__header.cosmology
        )

        return Dataset(
            self.__handler,
            self.__header,
            self.__unit_transformations,
            new_transformations,
        )
