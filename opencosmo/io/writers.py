from typing import Callable, Optional

import h5py
import numpy as np

from opencosmo.dataset.index import DataIndex
from opencosmo.header import OpenCosmoHeader
from opencosmo.io import protocols as iop
from opencosmo.mpi import get_comm_world

"""
Writers work in tandem with schemas to create new files. All schemas must have
an into_writer method, which returns a writer that can be used to put
data into the new file.

Schemas are responsible for validating and building the file structure 
as well as allocating space.  As a result, writers ASSUME the correct structure exists, 
and that all the datasets have the correct size, datatype, etc.
"""


def write_index(
    input_ds: h5py.Dataset,
    output_ds: h5py.Dataset,
    index: DataIndex,
    offset: int = 0,
    updater: Optional[Callable[[np.ndarray], np.ndarray]] = None,
):
    """
    Helper function to take elements from one h5py.Dataset using an index
    and put it in a different one.
    """
    if len(index) == 0:
        raise ValueError("No indices provided to write")
    data = index.get_data(input_ds)
    if updater is not None:
        data = updater(data)

    data = data.astype(input_ds.dtype)

    if output_ds.file.driver == "mpio":
        with output_ds.collective:
            output_ds[offset : offset + len(data)] = data
    else:
        output_ds[offset : offset + len(data)] = data


class FileWriter:
    """
    Root writer for a file. Pretty much just calls the child writers.
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
    Writes collections to a file or grous. Also pretty much just calls
    the child writers.
    """

    def __init__(
        self,
        children: dict[str, iop.DataWriter],
        header: Optional[OpenCosmoHeader] = None,
    ):
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

    def __init__(
        self,
        columns: dict[str, "ColumnWriter"],
        links: dict[str, "LinkWriter"] = {},
        header: Optional[OpenCosmoHeader] = None,
    ):
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
    Writes a single column in a dataset. This is the only writer that actually moves
    real data around.
    """

    def __init__(
        self, name: str, index: DataIndex, source: h5py.Dataset, offset: int = 0
    ):
        self.name = name
        self.source = source
        self.index = index
        self.offset = offset

    def write(
        self,
        group: h5py.Group,
        updater: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        ds = group[self.name]
        write_index(self.source, ds, self.index, self.offset, updater)
        for name, val in self.source.attrs.items():
            ds.attrs[name] = val


def idx_link_updater(input: np.ndarray, offset: int = 0) -> np.ndarray:
    output = np.full(len(input), -1)
    good = input >= 0
    output[good] = np.arange(sum(good)) + offset
    return output


def make_idx_link_updater(input: ColumnWriter) -> Callable[[np.ndarray], np.ndarray]:
    """
    Helper function to update data from an 1-to-1 index
    link.
    """
    arr = input.index.get_data(input.source)

    has_data = arr > 0
    offset = 0
    n_good = sum(has_data)
    if (comm := get_comm_world()) is not None:
        all_sizes = comm.allgather(n_good)
        offsets = np.insert(np.cumsum(all_sizes), 0, 0)
        offset = offsets[comm.Get_rank()]
    return lambda arr_: idx_link_updater(arr_, offset)


class IdxLinkWriter:
    """
    Writer for links between datasets, where each row in one dataset corresponds
    to a single row in the other.
    """

    def __init__(self, col_writer: ColumnWriter):
        self.writer = col_writer
        self.updater = make_idx_link_updater(self.writer)

    def write(self, group: h5py.Group):
        self.writer.write(group, self.updater)


def start_link_updater(sizes: np.ndarray, offset: int = 0) -> np.ndarray:
    cumulative_sizes = np.cumsum(sizes)

    new_starts = np.insert(cumulative_sizes, 0, 0)
    new_starts = new_starts[:-1] + offset
    return new_starts


def make_start_link_updater(
    size_writer: ColumnWriter,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Helper function to update the starts of a start-size
    link. Required to work this way so that we can write
    in an MPI context WITHOUT using parallel hdf5
    """
    sizes = size_writer.index.get_data(size_writer.source)
    cumulative_sizes = np.cumsum(sizes)
    if (comm := get_comm_world()) is not None:
        offsets = np.cumsum(comm.allgather(cumulative_sizes[-1]))
        offsets = np.insert(offsets, 0, 0)
        offset = offsets[comm.Get_rank()]
    else:
        offset = 0

    return lambda arr_: start_link_updater(arr_, offset)


class StartSizeLinkWriter:
    """
    Writer for links between datasets where each row in one datest
    corresponds to several rows in the other.
    """

    def __init__(self, start: ColumnWriter, size: ColumnWriter):
        self.start = start
        self.sizes = size
        self.updater = make_start_link_updater(size)

    def write(self, group: h5py.Group):
        self.sizes.write(group)
        new_sizes = self.sizes.index.get_data(self.sizes.source)
        self.start.write(group, lambda _: self.updater(new_sizes))


LinkWriter = IdxLinkWriter | StartSizeLinkWriter
