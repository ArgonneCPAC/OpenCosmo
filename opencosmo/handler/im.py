from copy import copy

import h5py
from astropy.table import Column, Table  # type: ignore

from opencosmo import transformations as t


class InMemoryHandler:
    """
    A handler for in-memory storage. All data will be loaded directly into memory.

    Reading a file is always done with OpenCosmo.read, which manages file opening
    and closing.
    """

    def __init__(self, file: h5py.File):
        colnames = file["data"].keys()
        self.__data = {colname: file["data"][colname][()] for colname in colnames}

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        return False

    def get_data(self, builders: dict = {}):
        """ """
        output = {}
        for column, builder in builders.items():
            output[column] = Column(copy(self.__data[column]))
            output[column] = builder.build(output[column])

        return Table(output)
