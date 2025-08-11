from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import h5py
import numpy as np

from opencosmo.index import DataIndex
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

    def make_indices(self, index: DataIndex) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if isinstance(self.link, tuple):
            start = index.get_data(self.link[0])
            size = index.get_data(self.link[1])
            valid_rows = size > 0
            starts = start[valid_rows]
            sizes = size[valid_rows]
            return starts, sizes
        else:
            indices_into_data = index.get_data(self.link)
            return indices_into_data[indices_into_data >= 0], None

    def make_schema(self, name: str, index: DataIndex) -> ios.LinkSchema:
        if isinstance(self.link, h5py.Dataset):
            return ios.IdxLinkSchema(name, index, self.link)
        else:
            return ios.StartSizeLinkSchema(name, index, *self.link)
