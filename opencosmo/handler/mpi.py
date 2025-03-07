from typing import Iterable, Optional, Tuple

import h5py
import numpy as np
from astropy.table import Column, Table  # type: ignore
from mpi4py import MPI


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

    def write(
        self,
        file: h5py.File,
        filter: np.ndarray,
        columns: Iterable[str],
        dataset_name="data",
    ) -> None:
        rank_output_length = np.sum(filter)
        all_output_lengths = self.__comm.gather(rank_output_length, root=0)

        # Determine the number of elements this rank is responsible for
        # writing
        rank_start = np.sum(all_output_lengths[:self.__comm.Get_rank()])
        rank_end = rank_start + rank_output_length

        full_output_length = np.sum(all_output_lengths)
        group = file.require_group(dataset_name)
        for column in columns:
            # This step has to be done by all ranks, per documentation
            group.create_dataset(column, (full_output_length,), dtype=self.__group[column].dtype)
        self.__comm.Barrier()

        for column in columns:
            data = self.__group[column][filter]
            group[column][rank_start:rank_end] = data
        self.__comm.Barrier()

    def get_data(
        self, builders: dict = {}, filter: Optional[np.ndarray] = None
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
            if filter is not None:
                data = data[filter]
            col = Column(data, name=column)
            output[column] = builder.build(col)

        if len(output) == 1:
            return next(iter(output.values()))
        return Table(output)
