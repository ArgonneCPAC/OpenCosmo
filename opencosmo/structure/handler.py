from __future__ import annotations

from typing import Iterable

import h5py
import numpy as np

import opencosmo as oc
from opencosmo.dataset.col import DerivedColumn
from opencosmo.index import ChunkedIndex, DataIndex, SimpleIndex
from opencosmo.io import schemas as ios


class LinkedDatasetHandler:
    """
    Links are currently only supported out-of-memory.
    """

    def __init__(
        self, link: h5py.Group | tuple[h5py.Group, h5py.Group], dataset: oc.Dataset
    ):
        self.link = link
        self.dataset = dataset

    def get_all_data(self) -> oc.Dataset:
        return self.dataset

    def get_dataset(self, index: DataIndex) -> oc.Dataset:
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

        return self.dataset.with_index(new_index)

    def select(self, columns: str | Iterable[str]) -> LinkedDatasetHandler:
        if isinstance(columns, str):
            columns = [columns]
        dataset = self.dataset.select(columns)
        return LinkedDatasetHandler(self.link, dataset)

    def with_units(self, convention: str) -> LinkedDatasetHandler:
        return LinkedDatasetHandler(
            self.link,
            self.dataset.with_units(convention),
        )

    def with_new_columns(self, **new_columns: DerivedColumn):
        return LinkedDatasetHandler(
            self.link, self.dataset.with_new_columns(**new_columns)
        )

    def make_schema(self, name: str, index: DataIndex) -> ios.LinkSchema:
        if isinstance(self.link, h5py.Dataset):
            return ios.IdxLinkSchema(name, index, self.link)
        else:
            return ios.StartSizeLinkSchema(name, index, *self.link)
