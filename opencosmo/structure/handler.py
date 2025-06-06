from __future__ import annotations

from typing import Iterable, Optional

import h5py
import numpy as np

import opencosmo as oc
from opencosmo.header import OpenCosmoHeader
from opencosmo.index import ChunkedIndex, DataIndex, SimpleIndex
from opencosmo.io import schemas as ios
from opencosmo.structure.builder import OomDatasetBuilder


class LinkedDatasetHandler:
    """
    Links are currently only supported out-of-memory.
    """

    def __init__(
        self,
        file: h5py.File | h5py.Group,
        link: h5py.Group | tuple[h5py.Group, h5py.Group],
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

    def select(self, columns: str | Iterable[str]) -> LinkedDatasetHandler:
        if isinstance(columns, str):
            columns = [columns]
        builder = self.builder.select(columns)
        return LinkedDatasetHandler(
            self.file,
            self.link,
            self.header,
            builder,
        )

    def with_units(self, convention: str) -> LinkedDatasetHandler:
        return LinkedDatasetHandler(
            self.file,
            self.link,
            self.header,
            self.builder.with_units(convention),
        )

    def make_schema(self, name: str, index: DataIndex) -> ios.LinkSchema:
        if isinstance(self.link, h5py.Dataset):
            return ios.IdxLinkSchema(name, index, self.link)
        else:
            return ios.StartSizeLinkSchema(name, index, *self.link)
