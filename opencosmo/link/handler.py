from __future__ import annotations
from typing import Protocol
from h5py import File, Group
import opencosmo as oc
from typing import Iterable, Optional
from opencosmo.header import OpenCosmoHeader, read_header
from opencosmo.transformations import units as u
from opencosmo.spatial import read_tree
from opencosmo.handler import OutOfMemoryHandler
from pathlib import Path
from collections import defaultdict

import numpy as np

       
def build_dataset(file: File | Group, indices: np.ndarray, header: OpenCosmoHeader) -> Dataset:
    tree = read_tree(file, header)
    builders, base_unit_transformations = u.get_default_unit_transformations(file, header)
    handler = OutOfMemoryHandler(file, tree=tree)
    return oc.Dataset(handler, header, builders, base_unit_transformations, indices)


class LinkHandler(Protocol):
    """
    A LinkHandler is responsible for handling linked datasets. Links are found
    in property files, and contain indexes into another dataset. The link handler
    is responsible for holding pointers to the linked files, and returning
    the associated data when requested (as a Dataset object).
    """

    def __init__(self, file: File | Group, links: Group | tuple[Group, Group], header: OpenCosmoHeader): ...
    def get_data(self, indices: int | np.ndarray) -> oc.Dataset: ...
    def write(self, data_group: Group, link_group: Group, indices: int | np.ndarray) -> None: ...


class OomLinkHandler:
    def __init__(self, file: File | Group, link: Group | tuple[Group, Group], header: OpenCosmoHeader = None):
        self.file = file
        self.link = link
        self.header = header

    def get_data(self, indices: np.ndarray) -> Optional[oc.Dataset]:
        if isinstance(self.link, tuple):
            start = self.link[0][indices]
            size = self.link[1][indices]
            valid_rows = size > 0
            start = start[valid_rows]
            size = size[valid_rows]
            if not start.size:
                return None
            indices = np.concatenate([np.arange(idx, idx + length) for idx, length in zip(start, size)])
        else:
            indices = self.link[indices]
            indices = indices[indices >= 0]
            if not indices.size:
                return None
        return build_dataset(self.file, indices, self.header)

    def write(self, file: File, name: str, link_group: Group, indices: np.ndarray):
        # Pack the indices
        if not isinstance(self.link, tuple): 
            new_idxs = np.arange(len(indices))
            link_group.create_dataset("sod_profile_idx", data=new_idxs, dtype=int)
        else:
            lengths = self.link[1][indices]
            new_starts = np.insert(np.cumsum(lengths), 0, 0)[:-1]
            link_group.create_dataset(f"{name}_start", data=new_starts, dtype=int)
            link_group.create_dataset(f"{name}_size", data=lengths, dtype=int)

        dataset = self.get_data(indices)
        dataset.write(file, name)
