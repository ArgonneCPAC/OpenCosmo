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
    starts = {}
    sizes = {}


    for level in range(max_level + 1):
        group = file[f"index/level_{level}"]
        level_starts = group["start"][()]
        level_sizes = group["size"][()]
        starts[level] = level_starts
        sizes[level] = level_sizes

    spatial_index = OctTreeIndex(header.simulation, max_level)
    return Tree(spatial_index, starts, sizes)


def write_tree(file: h5py.File, tree: Tree, dataset_name: str = "index"):
    tree.write(file, dataset_name)


class Tree:
    """
    The Tree handles the spatial indexing of the data. As of right now, it's only
    functionality is to read and write the spatial index. Later we will add actual
    spatial queries
    """

    def __init__(self, index: SpatialIndex, starts: dict[int], sizes: dict[int]):
        self.__index = index
        self.__starts = starts
        self.__sizes = sizes


    def apply_mask(self, mask: np.ndarray) -> Tree:
        """
        Given a boolean mask, create a new tree with slices adjusted to
        only include the elements where the mask is True. This is used
        when writing filtered datasets to file, or collecting.

        The mask will have the same shape as the original data.
        """

        if np.all(mask):
            return self
        output_starts = {}
        output_sizes = {}
        for level in self.__starts:
            start = self.__starts[level]
            size = self.__sizes[level]
            offsets = np.zeros_like(size)
            for i in range(len(start)):
                # Create a slice object for the current level
                s = slice(start[i], start[i] + size[i])
                slice_mask = mask[s]  # Apply the slice to the mask
                offsets[i] = np.sum(slice_mask)  # Count the number of True values
            level_starts = np.cumsum(np.insert(offsets, 0, 0))[:-1]  # Cumulative sum to get new starts
            level_sizes = offsets
            output_starts[level] = level_starts
            output_sizes[level] = level_sizes

        return Tree(self.__index, output_starts, output_sizes)



                # Apply the mask to get the new slice


    
        
    def write(self, file: h5py.File, dataset_name: str = "index"):
        """
        Write the tree to an HDF5 file. Note that this function
        is not responsible for applying masking. The routine calling this
        function should first create a new tree with apply_mask if
        necessary.
        """
        group = file.require_group(dataset_name)
        for level in self.__starts:
            level_group = group.require_group(f"level_{level}")
            level_group.create_dataset("start", data=self.__starts[level])
            level_group.create_dataset("size", data=self.__sizes[level])
