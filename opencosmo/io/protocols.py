from typing import Optional, Protocol, Any

import h5py

try:
    from mpi4py import MPI
except ImportError:
    MPI = None  # type: ignore


class DataSchema(Protocol):
    def insert(self, child: "DataSchema", path: str): ...
    def add_child(self, child: "DataSchema", child_id: Any): ...
    def allocate(self, group: h5py.File | h5py.Group): ...
    def verify(self): ...
    def into_writer(self, comm: Optional["MPI.Comm"]): ...


class DataWriter(Protocol):
    def write(self, group: h5py.File | h5py.Group): ...


class Writeable(Protocol):
    def make_schema(self) -> DataSchema: ...
