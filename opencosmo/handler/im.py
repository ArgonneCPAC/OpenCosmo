from copy import copy

import h5py
from astropy.table import Table  # type: ignore

from opencosmo import transformations as t


class InMemoryHandler:
    """
    A handler for in-memory storage. All data will be loaded directly into memory.

    Reading a file is always done with OpenCosmo.read, which manages file opening
    and closing.
    """

    def __init__(self, file: h5py.File):
        colnames = file["data"].keys()
        data = {colname: file["data"][colname][()] for colname in colnames}
        self.__data = Table(data)

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        return False

    def get_data(self, transformations: dict = {}):
        """ """
        table_transformations = transformations.get(t.TransformationType.TABLE, [])
        column_transformations = transformations.get(t.TransformationType.COLUMN, [])
        new_data = copy(self.__data)
        new_data = t.apply_column_transformations(new_data, column_transformations)
        new_data = t.apply_table_transformations(new_data, table_transformations)
        return new_data

    def select_columns(self, columns: list[str], transformations: dict = {}):
        new_data = self.__data[columns]
        # Problem: Transformations can rename columns...
        table_transformations = transformations.get(t.TransformationType.TABLE, [])
        column_transformations = transformations.get(t.TransformationType.COLUMN, [])
        new_data = t.apply_column_transformations(new_data, column_transformations)
        new_data = t.apply_table_transformations(new_data, table_transformations)
        return new_data
