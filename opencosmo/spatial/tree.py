from opencosmo.spatial.index import SpatialIndex
from opencosmo.spatial.octree import OctTreeIndex
from opencosmo.spatial.region import BoxRegion
import h5py
from opencosmo.header import OpenCosmoHeader
from typing import Iterable
from collections import OrderedDict


def read_tree(file: h5py, header: OpenCosmoHeader):
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




class Tree:
    def __init__(self, index: SpatialIndex, slices: dict[int, dict[int, slice]]):
        self.__index = index
        self.__slices = slices

    def query(self, region: BoxRegion) -> Iterable[slice]:
        overlaps = self.__index.get_regions(region)
        max_level = self.__index.max_level
        print(list(overlaps))
        max_level_index = self.__slices[max_level]
        return [max_level_index[overlap] for overlap in overlaps]
