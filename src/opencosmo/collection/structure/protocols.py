from typing import Iterable, Optional, Protocol, Self

import h5py

from opencosmo.dataset import Dataset
from opencosmo.header import OpenCosmoHeader
from opencosmo.index import DataIndex


class LinkHandler(Protocol):
    """
    A LinkHandler is responsible for handling linked datasets. Links are found
    in property files, and contain indexes into another dataset. For example, a
    halo properties file will contain links to a halo particles file. Each halo
    in the properties file will have a corresponding range of indexes that contain
    the associated particles in the particles file.

    The link handler is responsible for reading data and instatiating datasets
    that contain the linked data for the given object. There will be one link
    handler for each linked dataset in the properties file. This potentially
    means there will be multiple pointers to a single particle file, for example.
    """

    def __init__(
        self,
        file: h5py.File | h5py.Group,
        link: h5py.Group | tuple[h5py.Group, h5py.Group],
        header: OpenCosmoHeader,
        builder: Optional["DatasetBuilder"] = None,
        **kwargs,
    ):
        """
        Initialize the LinkHandler with the file, link, header, and optional builder.
        The builder is used to build the dataset from the file.
        """
        pass

    def get_data(self, index: DataIndex) -> Dataset:
        """
        Given a index or a set of indices, return the data from the linked dataset
        that corresponds to the halo/galaxy at that index in the properties file.
        Sometimes the linked dataset will not have data for that object, in which
        a zero-length dataset will be returned.
        """
        pass

    def get_all_data(self) -> Dataset:
        """
        Return all the data from the linked dataset.
        """
        pass

    def prep_write(
        self,
        data_group: h5py.Group,
        link_group: h5py.Group,
        name: str,
        index: DataIndex,
    ) -> None:
        """
        Write the linked data for the given indices to data_group.
        This function will then update the links to be consistent with the newly
        written data, and write the updated links to link_group.
        """
        pass

    def select(self, columns: str | Iterable[str]) -> Self:
        """
        Return a new LinkHandler that only contains the data for the given indices.
        """
        pass

    def with_units(self, convention: str) -> Self:
        """
        Return a new LinkHandler that uses the given unit convention.
        """
        pass


class DatasetBuilder(Protocol):
    """
    A DatasetBuilder is responsible for building a dataset from a file. It
    contains the logic for selecting columns and applying transformations to
    the data.
    """

    def with_units(self, convention: str) -> Self:
        pass

    def select(self, selected: str | Iterable[str]) -> Self:
        pass

    def build(
        self,
        file: h5py.File | h5py.Group,
        header: OpenCosmoHeader,
        index: Optional[DataIndex] = None,
    ) -> Dataset:
        pass
