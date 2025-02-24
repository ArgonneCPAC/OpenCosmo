from functools import property

from opencosmo.handler import OpenCosmoDataHandler
from opencosmo.header import OpenCosmoHeader


class OpenCosmoDataset:
    def __init__(self, handler: OpenCosmoDataHandler, header: OpenCosmoHeader):
        self.__header = header
        self.__handler = handler

    @property
    def cosmology(self):
        return self.__header.cosmology
