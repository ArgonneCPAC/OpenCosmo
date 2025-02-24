from __future__ import annotations

import h5py

from opencosmo.file import file_reader
from opencosmo.handler import InMemoryHandler, OpenCosmoDataHandler
from opencosmo.header import OpenCosmoHeader, read_header


@file_reader
def read(file: h5py.File) -> OpenCosmoDataset:
    """
    Read a dataset from a file into memory.

    You should use this function if the data are small enough that having
    a copy of it (or a few copies of it) in memory is not a problem. For
    larger datasets, use opencosmo.open

    Parameters
    ----------
    file : str or pathlib.Path
        The path to the file to read.

    Returns
    -------
    dataset : OpenCosmoDataset
        The dataset read from the file.

    """
    header = read_header(file)
    handler = InMemoryHandler(file)
    return OpenCosmoDataset(handler, header)


class OpenCosmoDataset:
    def __init__(self, handler: OpenCosmoDataHandler, header: OpenCosmoHeader):
        self.__header = header
        self.__handler = handler

    def __enter__(self):
        return self.__handler.__enter__()

    def __exit__(self, *exc_details):
        return self.__handler.__exit__(*exc_details)

    @property
    def cosmology(self):
        return self.__header.cosmology

    @property
    def data(self):
        return self.__handler.get_data()
