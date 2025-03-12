from typing import Iterable, Optional, Tuple

import h5py
import numpy as np
from astropy.table import Column, Table  # type: ignore
from mpi4py import MPI

from warnings import warn
from opencosmo.handler import InMemoryHandler


class MPIHandler:
    """
    A handler for reading and writing data in an MPI context.
    """

    def __init__(self, file: h5py.File, group: str = "data", comm=MPI.COMM_WORLD):
        self.__file = file
        self.__group = file[group]
        self.__columns = list(self.__group.keys())
        self.__comm = comm

    def elem_range(self) -> Tuple[int, int]:
        """
        The full dataset will be split into equal parts by rank.
        """
        nranks = self.__comm.Get_size()
        rank = self.__comm.Get_rank()
        n = self.__group[self.__columns[0]].size

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

        masks = self.__comm.allgather(mask)
        file_path = self.__file.filename
        output_mask = np.concatenate(masks)
        with h5py.File(file_path, "r") as file:
            return InMemoryHandler(file, columns=columns, mask=output_mask)



    def write(
        self,
        file: h5py.File,
        mask: np.ndarray,
        columns: Iterable[str],
        dataset_name="data",
    ) -> None:
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
        group = file.require_group(dataset_name)
        for column in columns:
            # This step has to be done by all ranks, per documentation
            group.create_dataset(
                column, (full_output_length,), dtype=self.__group[column].dtype
            )

        self.__comm.Barrier()

        for column in columns:
            data = self.__group[column][rank_range[0] : rank_range[1]][mask]

            group[column][rank_start:rank_end] = data

        self.__comm.Barrier()

    def get_data(
        self, builders: dict = {}, mask: Optional[np.ndarray] = None
    ) -> Column | Table:
        """
        Get data from the file in the range for this rank.
        """
        if self.__group is None:
            raise ValueError("This file has already been closed")
        output = {}
        range_ = self.elem_range()
        for column, builder in builders.items():
            data = self.__group[column][range_[0] : range_[1]]
            if mask is not None:
                data = data[mask]
            col = Column(data, name=column)
            output[column] = builder.build(col)

        if len(output) == 1:
            return next(iter(output.values()))
        return Table(output)

    def update_mask(self, n: int, strategy: str, mask: np.ndarray) -> np.ndarray:
        """
        This is the tricky one. We need to update the mask based on the amount of
        data in ALL the ranks.

        masks are localized to each rank. For "start" and "end" it's just a matter of
        figuring out how many elements each rank is responsible for. For "random" we
        need to be more clever.
        """
        n_requested = self.__comm.allgather(n)
        # Needs to be the same on all ranks
        if len(set(n_requested)) > 1:
            warn(
                "Requested different amounts of data on different ranks. Using the "
                "value from rank 0."
            )
            n = n_requested[0]

        rank_length = np.sum(mask)
        rank_lengths = self.__comm.allgather(rank_length)

        total_length = np.sum(rank_lengths)
        if n > total_length:
            # All ranks crash
            raise ValueError(
                f"Requested {n} elements, but only {total_length} are available."
            )

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
