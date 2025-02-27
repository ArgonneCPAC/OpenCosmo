from __future__ import annotations

import h5py

from opencosmo.file import file_reader
from opencosmo.handler import InMemoryHandler, OpenCosmoDataHandler
from opencosmo.header import OpenCosmoHeader, read_header
from opencosmo.transformations import generate_transformations
from opencosmo.transformations import units as u


@file_reader
def read(file: h5py.File, units: str = "comoving") -> OpenCosmoDataset:
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
    dataset : OpenCosmoDataset
        The dataset read from the file.

    """
    header = read_header(file)
    handler = InMemoryHandler(file)
    base_unit_transformations, transformations = u.get_unit_transformations(
        file["data"], header.cosmology, units
    )

    # merge the dictionaries

    return OpenCosmoDataset(handler, header, base_unit_transformations, transformations)


class OpenCosmoDataset:
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
        return self.__handler.__enter__()

    def __exit__(self, *exc_details):
        return self.__handler.__exit__(*exc_details)

    @property
    def cosmology(self):
        return self.__header.cosmology

    @property
    def data(self):
        return self.__handler.get_data(transformations=self.__transformations)

    def with_convention(self, convention: str) -> OpenCosmoDataset:
        """
        Get a new dataset with a different unit convention.

        Parameters
        ----------
        convention : str
            The unit convention to use. One of "physical", "comoving",
            "scalefree", or "unitless".

        Returns
        -------
        dataset : OpenCosmoDataset
            The new dataset with the requested unit convention.

        """
        new_transformations = u.get_unit_transition_transformations(
            convention, self.__unit_transformations, self.__header.cosmology
        )

        return OpenCosmoDataset(
            self.__handler,
            self.__header,
            self.__unit_transformations,
            new_transformations,
        )
