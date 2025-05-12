from typing import Iterable, Optional, Protocol, TypeVar, Callable

import h5py

import numpy as np
from opencosmo.dataset.index import DataIndex
from opencosmo.io import schemas as ios
from opencosmo.io import protocols as iop
from opencosmo.header import OpenCosmoHeader


try:
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() == 1:
        raise ImportError()
except ImportError:
    MPI = None


"""
Writers work in tandem with schemas to create new files. All schemas must have
an into_writer method, which returns a writer that can be used to put
data into the new file.

Because schemas allocate space in a file, the correct structure is always
assumed to be there by the writer. If the writer itself fails, it is probably either
a problem with the schema or there was a mixup in naming.
"""
def write_index(
    input_ds: h5py.Dataset,
    output_ds: h5py.Dataset,
    index: DataIndex,
    offset: int = 0,
    updater: Optional[Callable[[np.ndarray], np.ndarray]] = None
):
    if len(index) == 0:
        raise ValueError("No indices provided to write")
    data = index.get_data(input_ds)
    if updater is not None:
        data = updater(data)

    data = data.astype(input_ds.dtype)


    if MPI is not None:
        with output_ds.collective:
            output_ds[offset:offset+len(data)] = data

    else:
        output_ds[offset:offset+len(data)] = data


class FileWriter:
    """
    Root writer for a file.
    """
    def __init__(self, children: dict[str, iop.DataWriter]): 
        self.children = children

    def write(self, file: h5py.File):
        if len(self.children) == 1:
            ds = next(iter(self.children.values()))
            return ds.write(file)
        for name, dataset in self.children.items():
            dataset.write(file[name])



class CollectionWriter:
    """
    Writes collections to a file or group
    """
    def __init__(self, children: dict[str, iop.DataWriter], header: Optional[OpenCosmoHeader] = None):
        self.children = children
        self.header = header

    def write(self, file: h5py.File | h5py.Group):
        child_names = list(self.children.keys())
        child_names.sort()
        for name in child_names:
            self.children[name].write(file[name])

        if self.header is not None:
            self.header.write(file)


class DatasetWriter:
    """
    Writes datasets to a file or group.
    """
    def __init__(self, columns: dict[str, "ColumnWriter"], links: dict = [], header: Optional[OpenCosmoHeader] = None):
        self.columns = columns
        self.header = header
        self.links = links

    def write(self, group: h5py.Group):
        data_group = group["data"]
        names = list(self.columns.keys())
        names.sort()
        for colname in names:
            self.columns[colname].write(data_group)


        if self.links:
            link_group = group["data_linked"]
            link_names = list(self.links.keys())
            link_names.sort()
            for name in link_names:
                self.links[name].write(link_group)

        if self.header is not None:
            self.header.write(group)


class ColumnWriter:
    """
    Writes a single column in a dataset
    """
    def __init__(self, name: str, index: DataIndex, source: h5py.Dataset, offset: int = 0):
        self.name = name
        self.source = source
        self.index = index
        self.offset = offset

    def write(self, group: h5py.Group, updater: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        ds = group[self.name]
        write_index(self.source, ds, self.index, self.offset, updater)
        for name, val in self.source.attrs.items():
            ds.attrs[name] = val

def idx_link_updater(input: np.ndarray) -> np.ndarray:

    output = np.full(len(input), -1)
    has_data = input > 0
    offset = 0
    n_good = sum(has_data)
    if MPI is not None:
        all_sizes = MPI.COMM_WORLD.allgather(n_good)
        offsets = np.insert(np.cumsum(all_sizes), 0, 0)
        offset = offsets[MPI.COMM_WORLD.Get_rank()]
    output[has_data] = np.arange(sum(has_data)) + offset
    return output


class IdxLinkWriter:
    """
    Writer for links between datasets, where each row in one dataset corresponds
    to a single row in the other.
    """
    def __init__(self, col_writer: ColumnWriter):
        self.writer = col_writer


    def write(self, group: h5py.Group):
        self.writer.write(group, idx_link_updater)

def start_link_updater(sizes: np.ndarray):
    cumulative_sizes = np.cumsum(sizes)
    if MPI is not None:
        offsets = np.cumsum(MPI.COMM_WORLD.allgather(cumulative_sizes[-1]))
        offsets = np.insert(offsets, 0, 0)
        offset = offsets[MPI.COMM_WORLD.Get_rank()]
    else:
        offset = 0

    new_starts = np.insert(cumulative_sizes, 0, 0)
    new_starts = new_starts[:-1] + offset
    return new_starts

class StartSizeLinkWriter:
    """
    Writer for links between datasets where each row in one datest
    corresponds to several rows in the other.
    """
    def __init__(self, start: ColumnWriter, size: ColumnWriter):
        self.start = start
        self.sizes = size

    def write(self, group: h5py.Group):
        self.sizes.write(group)
        new_sizes = self.sizes.index.get_data(self.sizes.source)
        self.start.write(group, lambda _: start_link_updater(new_sizes))




        


