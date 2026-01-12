from __future__ import annotations

from enum import Enum
from functools import reduce
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

import numpy as np

from opencosmo.index import get_data, get_length

if TYPE_CHECKING:
    import h5py
    from numpy.typing import DTypeLike

    from opencosmo.index import DataIndex
    from opencosmo.mpi import MPI


class ColumnCombineStrategy(Enum):
    SUM = "sum"
    CONCAT = "concat"


class ColumnSource(Protocol):
    def __len__(self) -> int: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> DTypeLike: ...
    @property
    def data(self) -> np.ndarray: ...


class ColumnWriter:
    def __init__(
        self,
        column_sources: list[ColumnSource],
        combine_strategy: ColumnCombineStrategy,
        attrs: dict[str, Any] = {},
    ):
        self.__sources = column_sources
        self.__combine_strategy = combine_strategy
        self.__attrs = attrs
        dtypes = set(map(lambda source: source.dtype, self.__sources))
        if len(dtypes) > 1:
            raise ValueError("A single column can not have multiple data types!")
        self.__dtype = dtypes.pop()
        self.__transformation: Optional[Callable] = None

    @classmethod
    def from_numpy_array(
        cls,
        data: np.ndarray,
        strategy: ColumnCombineStrategy = ColumnCombineStrategy.CONCAT,
        attrs: dict[str, Any] = {},
    ):
        source = NumpySource(data)
        return ColumnWriter([source], strategy, attrs)

    @classmethod
    def from_h5_dataset(
        cls,
        dataset: h5py.Dataset,
        index: DataIndex,
        strategy: ColumnCombineStrategy = ColumnCombineStrategy.CONCAT,
        attrs: dict[str, Any] = {},
    ):
        source = Hdf5Source(dataset, index)
        return ColumnWriter([source], strategy, attrs)

    def combine(self, others: list[ColumnWriter]):
        new_sources = reduce(
            lambda acc, other: acc + other.__sources, others, self.__sources
        )
        new_sources = list(filter(lambda s: len(s) > 0, new_sources))
        return ColumnWriter(new_sources, self.combine_strategy, self.__attrs)

    def set_transformation(
        self, transformation: Callable[[np.ndarray, Optional[MPI.Comm]], np.ndarray]
    ):
        if self.__transformation is not None:
            raise ValueError(
                "A transformation can only be set on a column writer a single time!"
            )
        self.__transformation = transformation

    def __len__(self):
        match self.combine_strategy:
            case ColumnCombineStrategy.CONCAT:
                return sum(len(source) for source in self.__sources)
            case ColumnCombineStrategy.SUM:
                return len(self.__sources[0])

    @property
    def has_transformation(self):
        return self.__transformation is not None

    @property
    def shape(self):
        shape = self.__sources[0].shape
        return (len(self),) + shape[1:]

    @property
    def combine_strategy(self) -> ColumnCombineStrategy:
        return self.__combine_strategy

    @property
    def dtype(self) -> DTypeLike:
        return self.__dtype

    @property
    def attrs(self) -> dict[str, Any]:
        return self.__attrs

    @property
    def data(self) -> np.ndarray:
        return self.get_data()

    def get_data(self, comm: Optional[MPI.Comm] = None):
        match self.combine_strategy:
            case ColumnCombineStrategy.CONCAT:
                data = np.concatenate([source.data for source in self.__sources])
            case ColumnCombineStrategy.SUM:
                data = np.vstack([source.data for source in self.__sources]).sum(axis=0)
        if self.__transformation is not None:
            data = self.__transformation(data, comm=comm)  # type: ignore
        return data


class Hdf5Source:
    def __init__(self, h5ds: h5py.Dataset, index: DataIndex):
        self.__source = h5ds
        self.__index = index

    def __len__(self):
        return get_length(self.__index)

    @property
    def shape(self):
        return (len(self),) + self.__source.shape[1:]

    @property
    def dtype(self):
        return self.__source.dtype

    @property
    def data(self):
        data = get_data(self.__source, self.__index)
        return data


class NumpySource:
    def __init__(self, arr: np.ndarray):
        self.__source = arr

    def __len__(self):
        return len(self.__source)

    @property
    def shape(self):
        return self.__source.shape

    @property
    def dtype(self):
        return self.__source.dtype

    @property
    def data(self):
        return self.__source
