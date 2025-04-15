from typing import Iterable, Optional, Tuple
from warnings import warn

import h5py
import numpy as np
from astropy.table import Column, Table  # type: ignore
from mpi4py import MPI

from opencosmo.dataset.index import DataIndex
from opencosmo.file import get_data_structure
from opencosmo.handler import InMemoryHandler
from opencosmo.spatial.tree import Tree


def partition(comm: MPI.Comm, length: int) -> Tuple[int, int]:
    nranks = comm.Get_size()
    rank = comm.Get_rank()
    if rank == nranks - 1:
        start = rank * (length // nranks)
        size = length - start
        return (start, size)

    start = rank * (length // nranks)
    end = (rank + 1) * (length // nranks)
    size = end - start
    return (start, size)


def verify_input(comm: MPI.Comm, require: Iterable[str] = [], **kwargs) -> dict:
    """
    Verify that the input is the same on all ranks.

    If not, use the value from rank 0 if require is false,
    otherwise raise an error.
    """
    output = {}
    for key, value in kwargs.items():
        values = comm.allgather(value)

        if isinstance(value, Iterable):
            sets = [frozenset(v) for v in values]
            if len(set(sets)) > 1:
                if key in require:
                    raise ValueError(
                        f"Requested different values for {key} on different ranks."
                    )
                else:
                    warn(f"Requested different values for {key} on different ranks.")
        elif len(set(values)) > 1:
            if key in require:
                raise ValueError(
                    f"Requested different values for {key} on different ranks."
                )
            else:
                warn(f"Requested different values for {key} on different ranks.")
        output[key] = values[0]
    return output


class MPIHandler:
    """
    A handler for reading and writing data in an MPI context.
    """

    def __init__(
        self,
        file: h5py.File,
        tree: Optional[Tree] = None,
        group_name: Optional[str] = None,
        comm=MPI.COMM_WORLD,
    ):
        self.__file = file
        self.__group_name = group_name
        if group_name is None:
            self.__group = file["data"]
        else:
            self.__group = file[f"{group_name}/data"]
        self.__columns = get_data_structure(self.__group)
        self.__comm = comm
        self.__tree = tree

    def __len__(self) -> int:
        return next(iter(self.__group.values())).shape[0]

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        self.__group = None
        self.__columns = None
        return self.__file.close()

    def collect(self, columns: Iterable[str], index: DataIndex) -> InMemoryHandler:
        # concatenate the masks from all ranks
        columns = list(columns)
        columns = verify_input(comm=self.__comm, columns=columns)["columns"]

        all_indices = self.__comm.allgather(index)
        file_path = self.__file.filename
        all_indices = all_indices[0].concatenate(*all_indices[1:])
        with h5py.File(file_path, "r") as file:
            return InMemoryHandler(
                file,
                tree=self.__tree,
                columns=columns,
                index=all_indices,
                group_name=self.__group_name,
            )

    def write(
        self,
        file: h5py.File,
        index: DataIndex,
        columns: Iterable[str],
        dataset_name: Optional[str] = None,
        selected: Optional[np.ndarray] = None,
    ) -> None:
        columns = list(columns)
        input_ = verify_input(
            comm=self.__comm,
            columns=columns,
            dataset_name=dataset_name,
            require=["dataset_name"],
        )
        columns = input_["columns"]

        # indices = redistribute_indices(indices, rank_range)

        rank_output_length = len(index)

        all_output_lengths = self.__comm.allgather(rank_output_length)

        rank = self.__comm.Get_rank()

        # Determine the number of elements this rank is responsible for
        # writing
        if not rank:
            rank_start = 0
        else:
            rank_start = np.sum(all_output_lengths[:rank])

        rank_end = rank_start + rank_output_length

        full_output_length = np.sum(all_output_lengths)
        if dataset_name is None:
            group = file
        else:
            group = file.require_group(dataset_name)
        data_group = group.create_group("data")

        for column in columns:
            # This step has to be done by all ranks, per documentation
            shape = (full_output_length,) + self.__group[column].shape[1:]
            data_group.create_dataset(column, shape, dtype=self.__group[column].dtype)
            if self.__columns[column] is not None:
                data_group[column].attrs["unit"] = self.__columns[column]

        self.__comm.Barrier()

        if rank_output_length != 0:
            for column in columns:
                data = index.get_data(self.__group[column])
                data_group[column][rank_start:rank_end] = data

        mask = np.zeros(len(self), dtype=bool)
        mask = index.set_data(mask, True)
        if self.__tree is not None:
            new_tree = self.__tree.apply_mask(mask, self.__comm, index.range())
            new_tree.write(group)  # type: ignore

        self.__comm.Barrier()

    def get_data(self, builders: dict, index: DataIndex) -> Column | Table:
        """
        Get data from the file in the range for this rank.
        """
        builder_keys = list(builders.keys())
        if self.__group is None:
            raise ValueError("This file has already been closed")

        output = {}

        for column in builder_keys:
            col = Column(index.get_data(self.__group[column]))
            output[column] = builders[column].build(col)
        if len(output) == 1:
            return next(iter(output.values()))
        return Table(output)

    def take_range(self, start: int, end: int, indices: np.ndarray) -> np.ndarray:
        if start < 0 or end > len(indices):
            raise ValueError("Requested range is not within the rank's range.")

        return indices[start:end]
