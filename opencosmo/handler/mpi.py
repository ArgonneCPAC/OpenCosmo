from typing import Iterable, Optional, Tuple
from warnings import warn

import h5py
import numpy as np
from astropy.table import Column, Table  # type: ignore
from mpi4py import MPI

from opencosmo.file import get_data_structure
from opencosmo.handler import InMemoryHandler
from opencosmo.spatial.tree import Tree


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
        tree: Tree,
        group: Optional[str] = None,
        comm=MPI.COMM_WORLD,
    ):
        self.__file = file
        self.__group_name = group
        if group is None:
            self.__group = file["data"]
        else:
            self.__group = file[f"{group}/data"]
        self.__columns = get_data_structure(self.__group)
        self.__comm = comm
        self.__tree = tree

    def elem_range(self) -> Tuple[int, int]:
        """
        The full dataset will be split into equal parts by rank.
        """
        nranks = self.__comm.Get_size()
        rank = self.__comm.Get_rank()
        n = self.__group[next(iter(self.__columns))].shape[0]

        if rank == nranks - 1:
            return (rank * (n // nranks), n)
        return (rank * (n // nranks), (rank + 1) * (n // nranks))

    def __len__(self) -> int:
        range_ = self.elem_range()
        return range_[1] - range_[0]

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        self.__group = None
        self.__columns = None
        return self.__file.close()

    def collect(self, columns: Iterable[str], mask: np.ndarray) -> InMemoryHandler:
        # concatenate the masks from all ranks
        columns = list(columns)
        columns = verify_input(comm=self.__comm, columns=columns)["columns"]

        masks = self.__comm.allgather(mask)
        file_path = self.__file.filename
        output_mask = np.concatenate(masks)
        if all(output_mask):
            output_mask = None
        with h5py.File(file_path, "r") as file:
            return InMemoryHandler(
                file,
                tree=self.__tree,
                columns=columns,
                mask=output_mask,
                group_name=self.__group_name,
            )

    def write(
        self,
        file: h5py.File,
        mask: np.ndarray,
        columns: Iterable[str],
        dataset_name: Optional[str] = None,
    ) -> None:
        columns = list(columns)
        input = verify_input(
            comm=self.__comm,
            columns=columns,
            dataset_name=dataset_name,
            fname=file.filename,
            require=["fname", "dataset_name"],
        )
        columns = input["columns"]

        rank_range = self.elem_range()
        rank_output_length = np.sum(mask)
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
            data_group.create_dataset(
                column, (full_output_length,), dtype=self.__group[column].dtype
            )
            if self.__columns[column] is not None:
                data_group[column].attrs["unit"] = self.__columns[column]

        self.__comm.Barrier()

        for column in columns:
            data = self.__group[column][rank_range[0] : rank_range[1]][()]
            data = data[mask]
            data_group[column][rank_start:rank_end] = data

        tree = self.__tree.apply_mask(mask)
        tree.write(group)

        self.__comm.Barrier()

    def get_data(
        self, builders: dict = {}, mask: Optional[np.ndarray] = None
    ) -> Column | Table:
        """
        Get data from the file in the range for this rank.
        """
        builder_keys = list(builders.keys())
        builder_keys = verify_input(comm=self.__comm, builder_keys=builder_keys)[
            "builder_keys"
        ]

        if self.__group is None:
            raise ValueError("This file has already been closed")
        output = {}
        range_ = self.elem_range()
        for column in builder_keys:
            builder = builders[column]
            data = self.__group[column][range_[0] : range_[1]]
            if mask is not None:
                data = data[mask]
            col = Column(data, name=column)
            output[column] = builder.build(col)

        if len(output) == 1:
            return next(iter(output.values()))
        return Table(output)

    def take_mask(self, n: int, strategy: str, mask: np.ndarray) -> np.ndarray:
        """
        This is the tricky one. We need to update the mask based on the amount of
        data in ALL the ranks.

        masks are localized to each rank. For "start" and "end" it's just a matter of
        figuring out how many elements each rank is responsible for. For "random" we
        need to be more clever.
        """
        n = verify_input(comm=self.__comm, n=n)["n"]

        rank_length = np.sum(mask)
        rank_lengths = self.__comm.allgather(rank_length)

        total_length = np.sum(rank_lengths)
        if n > total_length:
            # All ranks crash
            warn(f"Requested {n} elements, but only {total_length} are available.")
            n = total_length

        if self.__comm.Get_rank() == 0:
            if strategy == "random":
                indices = np.random.choice(total_length, n, replace=False)
                indices = np.sort(indices)
            elif strategy == "start":
                indices = np.arange(n)
            elif strategy == "end":
                indices = np.arange(total_length - n, total_length)
            # Distribute the indices to the ranks
        else:
            indices = None
        indices = self.__comm.bcast(indices, root=0)

        if indices is None:
            # Should not happen, but this is for mypy
            raise ValueError("Indices should not be None.")

        rank_start_index = self.__comm.Get_rank()
        if rank_start_index:
            rank_start_index = np.sum(rank_lengths[: self.__comm.Get_rank()])
        rank_end_index = rank_start_index + rank_length
        rank_indicies = indices[
            (indices >= rank_start_index) & (indices < rank_end_index)
        ]

        new_true_indices = np.where(mask)[0][rank_indicies - rank_start_index]
        new_mask = np.zeros_like(mask)
        new_mask[new_true_indices] = True

        return new_mask
