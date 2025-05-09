from __future__ import annotations

from typing import Iterable, Optional, Protocol

import numpy as np
from h5py import File, Group

import opencosmo as oc
from opencosmo.dataset.index import ChunkedIndex, DataIndex, SimpleIndex
from opencosmo.header import OpenCosmoHeader
from opencosmo.link.builder import DatasetBuilder, OomDatasetBuilder
from opencosmo.io import schemas as ios


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
        file: File | Group,
        link: Group | tuple[Group, Group],
        header: OpenCosmoHeader,
        builder: Optional[DatasetBuilder] = None,
        **kwargs,
    ):
        """
        Initialize the LinkHandler with the file, link, header, and optional builder.
        The builder is used to build the dataset from the file.
        """
        pass

    def get_data(self, index: DataIndex) -> oc.Dataset:
        """
        Given a index or a set of indices, return the data from the linked dataset
        that corresponds to the halo/galaxy at that index in the properties file.
        Sometimes the linked dataset will not have data for that object, in which
        a zero-length dataset will be returned.
        """
        pass

    def get_all_data(self) -> oc.Dataset:
        """
        Return all the data from the linked dataset.
        """
        pass

    def prep_write(
        self, data_group: Group, link_group: Group, name: str, index: DataIndex
    ) -> None:
        """
        Write the linked data for the given indices to data_group.
        This function will then update the links to be consistent with the newly
        written data, and write the updated links to link_group.
        """
        pass

    def select(self, columns: str | Iterable[str]) -> LinkHandler:
        """
        Return a new LinkHandler that only contains the data for the given indices.
        """
        pass

    def with_units(self, convention: str) -> LinkHandler:
        """
        Return a new LinkHandler that uses the given unit convention.
        """
        pass


class OomLinkHandler:
    """
    Links are currently only supported out-of-memory.
    """

    def __init__(
        self,
        file: File | Group,
        link: Group | tuple[Group, Group],
        header: OpenCosmoHeader,
        builder: Optional[OomDatasetBuilder] = None,
    ):
        self.file = file
        self.link = link
        self.header = header

        if builder is None:
            self.builder = OomDatasetBuilder(
                selected=None,
                unit_convention=None,
            )
        else:
            self.builder = builder

    def get_all_data(self) -> oc.Dataset:
        return self.builder.build(
            self.file,
            self.header,
        )

    def get_data(self, index: DataIndex) -> oc.Dataset:
        if isinstance(self.link, tuple):
            start = index.get_data(self.link[0])
            size = index.get_data(self.link[1])
            valid_rows = size > 0
            start = start[valid_rows]
            size = size[valid_rows]
            new_index: DataIndex
            if not start.size:
                new_index = SimpleIndex(np.array([], dtype=int))
            else:
                new_index = ChunkedIndex(start, size)
        else:
            indices_into_data = index.get_data(self.link)
            indices_into_data = indices_into_data[indices_into_data >= 0]
            new_index = SimpleIndex(indices_into_data)

        return self.builder.build(self.file, self.header, new_index)

    def select(self, columns: str | Iterable[str]) -> OomLinkHandler:
        if isinstance(columns, str):
            columns = [columns]
        builder = self.builder.select(columns)
        return OomLinkHandler(
            self.file,
            self.link,
            self.header,
            builder,
        )

    def with_units(self, convention: str) -> OomLinkHandler:
        return OomLinkHandler(
            self.file,
            self.link,
            self.header,
            self.builder.with_units(convention),
        )

    def make_schema(self, name: str, index: DataIndex) -> ios.LinkSchema:
        return ios.LinkSchema(name, index, self.link)



