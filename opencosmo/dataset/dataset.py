from functools import property

from opencosmo.handler import OpenCosmoDataHandler
from opencosmo.header import OpenCosmoHeader


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
