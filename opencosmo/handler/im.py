from pathlib import Path

from astropy.io.misc import read_table_hdf5


class InMemoryHandler:
    """
    A handler for in-memory storage. All data will be loaded directly into memory, and
    the file will be closed.
    """

    def __init__(self, file_path: Path):
        self.__path = file_path
        self.__data = read_table_hdf5(self.__path, path="/data")

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        return False

    def apply_filters(self, filters: dict = {}):
        data = self.__data
        for filter_ in filters:
            data = filter_(data)
        return data

    def apply_transformations(self, transformations: dict = {}):
        data = self.__data
        for transformation in transformations:
            data = transformation(data)
        self.__data = data

    def get_data(self, filters: dict = {}, transformations: dict = {}):
        data = self.apply_filters(filters)
        data = self.apply_transformations(transformations)
        return data
