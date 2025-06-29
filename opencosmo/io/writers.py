from typing import Callable, Mapping, Optional

import h5py
import numpy as np

from opencosmo.header import OpenCosmoHeader
from opencosmo.index import DataIndex
from opencosmo.io import protocols as iop

try:
    from mpi4py import MPI
except ImportError:
    MPI = None  # type: ignore

"""
Writers work in tandem with schemas to create new files. All schemas must have
an into_writer method, which returns a writer that can be used to put
data into the new file.

Schemas are responsible for validating and building the file structure 
as well as allocating space.  As a result, writers ASSUME the correct structure exists, 
and that all the datasets have the correct size, datatype, etc.
"""


def write_index(
    input_ds: h5py.Dataset | np.ndarray,
    output_ds: h5py.Dataset,
    index: DataIndex,
    offset: int = 0,
    updater: Optional[Callable[[np.ndarray], np.ndarray]] = None,
):
    """
    Helper function to take elements from one h5py.Dataset using an index
    and put it in a different one.
    """
    data = np.array([])
    if len(index) > 0:
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
    the child writers. May or may not recieve a header to write, depending
    on they type of collection.
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
    Writes datasets to a file or group. Datasets must have at least one column.
    If the datset is being written alone or as part of SimulationCollection, it will
    be responsible for writing a header.

    It may or may not have a spatial index. It also may or may not have links
    to other datasets.
    """

    def __init__(
        self,
        columns: dict[str, "ColumnWriter"],
        links: dict[str, "LinkWriter"] = {},
        spatial_index: Optional["SpatialIndexWriter"] = None,
        header: Optional[OpenCosmoHeader] = None,
    ):
        self.columns = columns
        self.header = header
        self.links = links
        self.spatial_index = spatial_index

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

        if self.spatial_index is not None:
            index_group = group["index"]
            self.spatial_index.write(index_group)

        if self.header is not None:
            self.header.write(group)


class ColumnWriter:
    """
    Writes a single column in a dataset. This is the only writer that actually moves
    real data around.
    """

    def __init__(
        self,
        name: str,
        index: DataIndex,
        source: h5py.Dataset,
        attrs: Mapping,
        offset: int = 0,
    ):
        self.name = name
        self.source = source
        self.index = index
        self.attrs = attrs
        self.offset = offset

    def write(
        self,
        group: h5py.Group,
        updater: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        ds = group[self.name]

        write_index(self.source, ds, self.index, self.offset, updater)

        for name, val in self.attrs.items():
            ds.attrs[name] = val

        ds.file.flush()


class SpatialIndexWriter:
    """
    Writer for spatial indices. Mostly responsible for calling its children
    """

    def __init__(self, levels: dict[int, "SpatialIndexLevelWriter"]):
        self.levels = levels

    def write(self, group: h5py.Group):
        for (
            level_num,
            writer,
        ) in self.levels.items():
            level_group = group[f"level_{level_num}"]
            writer.write(level_group)


class SpatialIndexLevelWriter:
    """
    Writer for writing a single level of the spatial index. If this operation is being
    performed in an MPI context, the spatial indices must be summed.
    """

    def __init__(
        self, start: ColumnWriter, size: ColumnWriter, comm: Optional["MPI.Comm"] = None
    ):
        self.start = start
        self.size = size
        self.updater = lambda data: sum_updater(data, comm)

    def write(self, group: h5py.Group):
        self.size.write(group, updater=self.updater)
        self.start.write(group, updater=self.updater)


def sum_updater(data: np.ndarray, comm: Optional["MPI.Comm"] = None):
    if comm is not None and comm.Get_size():
        recvbuf = np.zeros_like(data)
        comm.Allreduce(data, recvbuf, MPI.SUM)
        return recvbuf
    return data


class IdxLinkWriter:
    """
    Writer for links between datasets, where each row in one dataset corresponds
    to a single row in the other. When the dataset is filtered, this link must be
    updated.
    """

    def __init__(self, col_writer: ColumnWriter, comm: Optional["MPI.Comm"] = None):
        self.writer = col_writer
        self.updater = make_idx_link_updater(self.writer, comm)

    def write(self, group: h5py.Group):
        self.writer.write(group, self.updater)


def idx_link_updater(input: np.ndarray, offset: int = 0) -> np.ndarray:
    output = np.full(len(input), -1)
    good = input >= 0
    output[good] = np.arange(sum(good)) + offset
    return output


def make_idx_link_updater(
    input: ColumnWriter, comm: Optional["MPI.Comm"]
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Helper function to update data from a 1-to-1 index
    link.
    """
    arr = input.index.get_data(input.source)

    has_data = arr > 0
    offset = 0
    n_good = sum(has_data)
    if comm is not None:
        all_sizes = comm.allgather(n_good)
        offsets = np.insert(np.cumsum(all_sizes), 0, 0)
        offset = offsets[comm.Get_rank()]
    return lambda arr_: idx_link_updater(arr_, offset)


class StartSizeLinkWriter:
    """
    Writer for links between datasets where each row in one datest
    corresponds to several rows in the other.
    """

    def __init__(
        self, start: ColumnWriter, size: ColumnWriter, comm: Optional["MPI.Comm"] = None
    ):
        self.start = start
        self.sizes = size
        self.updater = make_start_link_updater(size, comm)

    def write(self, group: h5py.Group):
        self.sizes.write(group)
        new_sizes = self.sizes.index.get_data(self.sizes.source)
        self.start.write(group, lambda _: self.updater(new_sizes))


def start_link_updater(sizes: np.ndarray, offset: int = 0) -> np.ndarray:
    cumulative_sizes = np.cumsum(sizes)

    new_starts = np.insert(cumulative_sizes, 0, 0)
    new_starts = new_starts[:-1] + offset
    return new_starts


def make_start_link_updater(
    size_writer: ColumnWriter, comm: Optional["MPI.Comm"]
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Helper function to update the starts of a start-size
    link.
    """
    sizes = size_writer.index.get_data(size_writer.source)
    total_size = np.sum(sizes)
    if comm is not None:
        offsets = np.cumsum(comm.allgather(total_size), dtype=np.uint64)
        offsets = np.insert(offsets, 0, 0)
        offset = offsets[comm.Get_rank()]
    else:
        offset = 0

    return lambda arr_: start_link_updater(arr_, offset)


LinkWriter = IdxLinkWriter | StartSizeLinkWriter
