from __future__ import annotations

from typing import Optional, Protocol

import numpy as np
from h5py import File, Group

import opencosmo as oc
from opencosmo.handler import OutOfMemoryHandler
from opencosmo.header import OpenCosmoHeader
from opencosmo.spatial import read_tree
from opencosmo.transformations import units as u


def build_dataset(
    file: File | Group, header: OpenCosmoHeader, indices: Optional[np.ndarray] = None
) -> oc.Dataset:
    tree = read_tree(file, header)
    builders, base_unit_transformations = u.get_default_unit_transformations(
        file, header
    )
    handler = OutOfMemoryHandler(file, tree=tree)
    if indices is None:
        indices = np.arange(len(handler))
    return oc.Dataset(handler, header, builders, base_unit_transformations, indices)


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
        links: Group | tuple[Group, Group],
        header: OpenCosmoHeader,
        *args,
        **kwargs,
    ): ...
    def get_data(self, indices: int | np.ndarray) -> Optional[oc.Dataset]:
        """
        Given a index or a set of indices, return the data from the linked dataset
        that corresponds to the halo/galaxy at that index in the properties file.
        Sometimes the linked dataset will not have data for that object, in which
        case None should be returned.
        """
        pass
    def get_all_data(self) -> oc.Dataset: 
        """
        Return all the data from the linked dataset. 
        """
        pass
    def write(
        self, data_group: Group, link_group: Group, name: str, indices: int | np.ndarray
    ) -> None:
        """
        Write the linked data for the given indices to data_group.
        This function will then update the links to be consistent with the newly
        written data, and write the updated links to link_group.
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
    ):
        self.file = file
        self.link = link
        self.header = header

    def get_all_data(self) -> oc.Dataset:
        return build_dataset(self.file, self.header)

    def get_data(self, indices: int | np.ndarray) -> Optional[oc.Dataset]:
        if isinstance(indices, int):
            indices = np.array([indices], dtype=int)
        min_idx = np.min(indices)
        max_idx = np.max(indices)

        if isinstance(self.link, tuple):
            start = self.link[0][min_idx : max_idx + 1][indices - min_idx]
            size = self.link[1][min_idx : max_idx + 1][indices - min_idx]
            valid_rows = size > 0
            start = start[valid_rows]
            size = size[valid_rows]
            if not start.size:
                return None
            indices_into_data = np.concatenate(
                [np.arange(idx, idx + length) for idx, length in zip(start, size)]
            )
        else:
            indices_into_data = self.link[min_idx : max_idx + 1][indices - min_idx]
            indices_into_data = np.array(indices_into_data[indices_into_data >= 0])
            if not indices_into_data.size:
                return None
        return build_dataset(self.file, self.header, indices_into_data)

    def write(
        self, group: Group, link_group: Group, name: str, indices: int | np.ndarray
    ):
        if isinstance(indices, int):
            indices = np.array([indices])
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
        if dataset is not None:
            dataset.write(group, name)
