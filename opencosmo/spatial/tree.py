from __future__ import annotations

from typing import Optional

import h5py
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None  # type: ignore

from opencosmo.header import OpenCosmoHeader
from opencosmo.spatial.index import SpatialIndex
from opencosmo.spatial.octree import OctTreeIndex


def read_tree(file: h5py.File | h5py.Group, header: OpenCosmoHeader):
    """
    Read a tree from an HDF5 file and the associated
    header. The tree is just a mapping between a spatial
    index and a slice into the data.

    Note: The max level in the header may not actually match
    the max level in the file. When a large dataset is filtered down,
    we may reduce the tree level to save space in the output file.

    The max level in the header is the maximum level in the full
    dataset, so this is the HIGHEST it can be.
    """
    max_level = header.reformat.max_level
    starts = {}
    sizes = {}

    for level in range(max_level + 1):
        try:
            group = file[f"index/level_{level}"]
        except KeyError:
            break
        level_starts = group["start"][()]
        level_sizes = group["size"][()]
        starts[level] = level_starts
        sizes[level] = level_sizes

    spatial_index = OctTreeIndex(header.simulation, max_level)
    return Tree(spatial_index, starts, sizes)


def write_tree(file: h5py.File, tree: Tree, dataset_name: str = "index"):
    tree.write(file, dataset_name)


def apply_range_mask(
    mask: np.ndarray,
    range_: tuple[int, int],
    starts: dict[int, np.ndarray],
    sizes: dict[int, np.ndarray],
) -> dict[int, tuple[int, np.ndarray]]:
    """
    Given an index range, apply a mask of the same size to produces new sizes.
    """
    output_sizes = {}

    for level, st in starts.items():
        ends = st + sizes[level]
        # Not in range if the end is less than start, or the start is greater than end
        overlaps_mask = ~((st > range_[1]) | (ends < range_[0]))
        # The first start may be less thank the range start so
        first_start_index = int(np.argmax(overlaps_mask))
        st = st[overlaps_mask]
        st[0] = range_[0]
        st = st - range_[0]
        # Determine how many true values are in the mask in the ranges
        new_sizes = np.add.reduceat(mask, st)
        output_sizes[level] = (first_start_index, new_sizes)
    return output_sizes


def pack_masked_ranges(
    old_starts: dict[int, np.ndarray],
    new_sizes: list[dict[int, tuple[int, np.ndarray]]],
    min_level_size: int = 500,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Given a list of masked ranges, pack them into a new set of sizes.
    This is used when working with MPI, and allows us to avoid sending
    very large masks between ranks.

    For queries that return a small fraction of the data, we can end up
    writing a lot of zeros in the lower levels of the tree. So we can
    dynamically choose to stop writing levels when the average size of
    the level is below a certain threshold
    """
    output_starts = {}
    output_sizes = {}
    for level in new_sizes[0]:
        new_level_sizes = np.zeros_like(old_starts[level])
        new_start_info = [rm[level] for rm in new_sizes]
        for first_idx, sizes in new_start_info:
            new_level_sizes[first_idx : first_idx + len(sizes)] += sizes

        avg_size = np.mean(new_level_sizes[new_level_sizes > 0])
        if avg_size < min_level_size:
            break
        output_sizes[level] = new_level_sizes
        output_starts[level] = np.cumsum(np.insert(new_level_sizes, 0, 0))[:-1]

    return output_starts, output_sizes


class Tree:
    """
    The Tree handles the spatial indexing of the data. As of right now, it's only
    functionality is to read and write the spatial index. Later we will add actual
    spatial queries
    """

    def __init__(
        self,
        index: SpatialIndex,
        starts: dict[int, np.ndarray],
        sizes: dict[int, np.ndarray],
    ):
        self.__index = index
        self.__starts = starts
        self.__sizes = sizes

    def apply_mask(
        self,
        mask: np.ndarray,
        comm: Optional[MPI.Comm] = None,
        range_: Optional[tuple] = None,
    ) -> Tree:
        """
        Given a boolean mask, create a new tree with slices adjusted to
        only include the elements where the mask is True. This is used
        when writing filtered datasets to file, or collecting.

        The mask will have the same shape as the original data.
        """

        if comm is not None and range_ is not None:
            return self.__apply_rank_mask(mask, comm, range_)
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
            level_starts = np.cumsum(np.insert(offsets, 0, 0))[
                :-1
            ]  # Cumulative sum to get new starts
            level_sizes = offsets
            output_starts[level] = level_starts
            output_sizes[level] = level_sizes

        return Tree(self.__index, output_starts, output_sizes)

    def __apply_rank_mask(
        self, mask: np.ndarray, comm: MPI.Comm, range_: tuple[int, int]
    ) -> Tree:
        """
        Given a range and a mask, apply the mask to the tree. The mask
        will have the same shape as the original data.
        """
        new_sizes = apply_range_mask(mask, range_, self.__starts, self.__sizes)
        all_new_sizes = comm.allgather(new_sizes)
        output_starts, output_sizes = pack_masked_ranges(self.__starts, all_new_sizes)
        return Tree(self.__index, output_starts, output_sizes)

    def write(self, file: h5py.File, dataset_name: str = "index"):
        """
        Write the tree to an HDF5 file. Note that this function
        is not responsible for applying masking. The routine calling this
        funct
        MPI = None
        MPI = None
        MPI = Noneion should first create a new tree with apply_mask if
        necessary.
        """
        group = file.require_group(dataset_name)
        for level in self.__starts:
            level_group = group.require_group(f"level_{level}")
            level_group.create_dataset("start", data=self.__starts[level])
            level_group.create_dataset("size", data=self.__sizes[level])
