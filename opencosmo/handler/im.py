import h5py
from astropy.table import Table  # type: ignore


def apply_columns_transformations(data: Table, transformations: dict):
    for column in data.columns:
        col = data[column]
        for transformation in transformations:
            transformed_col = transformation(col)
            if transformed_col is not None:
                col = transformed_col
        if col is not None:
            data[column] = col

    return data


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
        columns_transformations = transformations.get("columns", [])
        return apply_columns_transformations(self.__data, columns_transformations)
