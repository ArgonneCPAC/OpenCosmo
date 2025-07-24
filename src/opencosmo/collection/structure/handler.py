from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np

from opencosmo.index import ChunkedIndex, DataIndex, SimpleIndex
from opencosmo.io import schemas as ios

if TYPE_CHECKING:
    pass


class LinkedDatasetHandler:
    """
    Links are currently only supported out-of-memory.
    """

    def __init__(
        self,
        link: h5py.Group | tuple[h5py.Group, h5py.Group],
    ):
        self.link = link

    def has_linked_data(self, index: DataIndex) -> np.ndarray:
        """
        Check which rows in this index actually have data
        """
        if isinstance(self.link, tuple):
            sizes = index.get_data(self.link[1])
            return sizes > 0

        else:
            rows = index.get_data(self.link)
            return rows != -1

    def make_index(self, index: DataIndex) -> DataIndex:
        if isinstance(self.link, tuple):
            start = index.get_data(self.link[0])
            size = index.get_data(self.link[1])
            valid_rows = size > 0
            start = start[valid_rows]
            size = size[valid_rows]
            if not start.size:
                return SimpleIndex(np.array([], dtype=int))
            else:
                return ChunkedIndex(start, size)
        else:
            indices_into_data = index.get_data(self.link)
            indices_into_data = indices_into_data[indices_into_data >= 0]
            return SimpleIndex(indices_into_data)

    def make_schema(self, name: str, index: DataIndex) -> ios.LinkSchema:
        if isinstance(self.link, h5py.Dataset):
            return ios.IdxLinkSchema(name, index, self.link)
        else:
            return ios.StartSizeLinkSchema(name, index, *self.link)
