from copy import copy

import h5py
from astropy.table import Table  # type: ignore

from opencosmo import transformations as t


class InMemoryHandler:
    """
    A handler for in-memory storage. All data will be loaded directly into memory, and
    the file will be closed.
    """

    def __init__(self, file: h5py.File):
        colnames = file["data"].keys()
        data = {colname: file["data"][colname][()] for colname in colnames}
        self.__data = Table(data)

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        return False

    def get_data(self, filters: dict = {}, transformations: dict = {}):
        """ """
        table_transformations = transformations.get("table", [])
        column_transformations = transformations.get("column", [])
        new_data = copy(self.__data)
        new_data = t.apply_column_transformations(new_data, column_transformations)
        new_data = t.apply_table_transformations(new_data, table_transformations)
        return new_data
