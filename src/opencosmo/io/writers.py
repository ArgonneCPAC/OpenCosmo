from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from opencosmo.index import SimpleIndex
from opencosmo.io.updaters import apply_updaters

if TYPE_CHECKING:
    import h5py

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
    input_ds: h5py.Dataset | np.ndarray | None,
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
    if len(index) > 0 and input_ds is not None:
        data = index.get_data(input_ds)
        if updater is not None:
            data = updater(data)

        data = data.astype(input_ds.dtype)

    if input_ds is not None:
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
        if len(self.children) == 1:
            next(iter(self.children.values())).write(file)
            return

        child_names = list(self.children.keys())
        child_names.sort()
        for name in child_names:
            self.children[name].write(file[name])


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
        columns: dict[str, dict[str, ColumnWriter]],
        comm: Optional[MPI.Comm] = None,
    ):
        self.columns = apply_updaters(columns, comm)
        self.comm = comm

    def write(self, group: h5py.Group):
        groupnames = list(self.columns.keys())
        groupnames.sort()
        for groupname in groupnames:
            colnames = list(self.columns[groupname].keys())
            colnames.sort()
            data_group = group[groupname]
            for colname in colnames:
                self.columns[groupname][colname].write(data_group)


class EmptyColumnWriter:
    def __init__(self, name: str):
        self.name = name

    def write(
        self,
        group: h5py.Group,
        updater: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        ds = group[self.name]

        write_index(None, ds, SimpleIndex.empty(), updater=updater)


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
        offset: int = 0,
        updater: Optional[Callable] = None,
    ):
        self.name = name
        self.source = source
        self.index = index
        self.offset = offset
        self.updater = updater
        self.data = None

    def write(
        self,
        group: h5py.Group,
    ):
        ds = group[self.name]

        write_index(self.source, ds, self.index, self.offset, self.updater)
