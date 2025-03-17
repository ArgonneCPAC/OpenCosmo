from __future__ import annotations

from collections import OrderedDict

import h5py
import numpy as np

from opencosmo.header import OpenCosmoHeader
from opencosmo.spatial.index import SpatialIndex
from opencosmo.spatial.octree import OctTreeIndex


def read_tree(file: h5py.File | h5py.Group, header: OpenCosmoHeader):
    """
    Read a tree from an HDF5 file and the associated
    header. The tree is just a mapping between a spatial
    index and a slice into the data.
    """
    max_level = header.reformat.max_level
    data_indices = OrderedDict()

    for level in range(max_level + 1):
        group = file[f"index/level_{level}"]
        starts = group["start"][()]
        sizes = group["size"][()]
        level_indices = {}
        for i, (start, size) in enumerate(zip(starts, sizes)):
            level_indices[i] = slice(start, start + size)
        data_indices[level] = level_indices

    spatial_index = OctTreeIndex(header.simulation, max_level)
    return Tree(spatial_index, data_indices)


def write_tree(file: h5py.File, tree: Tree, dataset_name: str = "index"):
    tree.write(file, dataset_name)


class Tree:
    """
    The Tree handles the spatial indexing of the data. As of right now, it's only
    functionality is to read and write the spatial index. Later we will add actual
    spatial queries
    """

    def __init__(self, index: SpatialIndex, slices: dict[int, dict[int, slice]]):
        self.__index = index
        self.__slices = slices

    def apply_mask(self, mask: np.ndarray) -> Tree:
        """
        Given a boolean mask, create a new tree with slices adjusted to
        only include the elements where the mask is True. This is used
        when writing filtered datasets to file, or collecting.

        The mask will have the same shape as the original data.
        """
        if np.all(mask):
            return self
        new_slices = {}
        for level, slices in self.__slices.items():
            lengths = [np.sum(mask[s]) for s in slices.values()]
            new_starts = np.cumsum([0] + lengths[:-1])
            new_slices[level] = {
                i: slice(new_starts[i], new_starts[i] + lengths[i])
                for i in range(len(lengths))
                if lengths[i] > 0
            }
        return Tree(self.__index, new_slices)

    def write(self, file: h5py.File, dataset_name: str = "index"):
        """
        Write the tree to an HDF5 file. Note that this function
        is not responsible for applying masking. The routine calling this
        function should first create a new tree with apply_mask if
        necessary.
        """
        group = file.require_group(dataset_name)
        for level, slices in self.__slices.items():
            level_group = group.require_group(f"level_{level}")
            start = np.array([s.start for s in slices.values()])
            size = np.array([s.stop - s.start for s in slices.values()])
            level_group.create_dataset("start", data=start)
            level_group.create_dataset("size", data=size)
