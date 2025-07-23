from copy import deepcopy
from typing import TypeGuard

import h5py
import numpy as np
from numpy.typing import NDArray

from opencosmo.index.protocols import DataIndex


class SimpleIndex:
    """
    An index of integers.
    """

    def __init__(self, index: NDArray[np.int_]) -> None:
        self.__index = np.sort(index)

    @classmethod
    def from_size(cls, size: int) -> "SimpleIndex":
        return SimpleIndex(np.arange(size))

    @classmethod
    def empty(cls):
        return SimpleIndex(np.array([], dtype=int))

    def __len__(self) -> int:
        return len(self.__index)

    def into_array(self) -> NDArray[np.int_]:
        return deepcopy(self.__index)

    def range(self) -> tuple[int, int]:
        """
        Guranteed to be sorted
        """
        if len(self) == 0:
            return 0, 0
        return self.__index[0], self.__index[-1]

    def into_mask(self):
        mask = np.zeros(self.__index[-1] + 1, dtype=bool)
        mask[self.__index] = True
        return mask

    def concatenate(self, *others: DataIndex) -> DataIndex:
        if len(others) == 0:
            return self

        indexes = np.concatenate([o.into_array() for o in others])
        indexes = np.concatenate((self.into_array(), indexes))
        return SimpleIndex(indexes)

    def n_in_range(
        self, start: NDArray[np.int_], size: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        if len(start) != len(size):
            raise ValueError("Start and size arrays must have the same length")
        if np.any(size < 0):
            raise ValueError("Sizes must greater than or equal to zero")
        if len(self) == 0:
            return np.zeros_like(start)

        ends = start + size
        start_idxs = np.searchsorted(self.__index, start, "left")
        end_idxs = np.searchsorted(self.__index, ends, "left")
        return end_idxs - start_idxs

    def set_data(self, data: np.ndarray, value: bool) -> np.ndarray:
        """
        Set the data at the index to the given value.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")

        data[self.__index] = value
        return data

    def take(self, n: int, at: str = "random") -> DataIndex:
        """
        Take n elements from the index.
        """
        if n > len(self):
            raise ValueError(f"Cannot take {n} elements from index of size {len(self)}")
        elif n == 0:
            return SimpleIndex.empty()

        if at == "random":
            return SimpleIndex(np.random.choice(self.__index, n, replace=False))
        elif at == "start":
            return SimpleIndex(self.__index[:n])
        elif at == "end":
            return SimpleIndex(self.__index[-n:])
        else:
            raise ValueError(f"Unknown value for 'at': {at}")

    def take_range(self, start: int, end: int) -> DataIndex:
        """
        Take a range of elements from the index.
        """
        if start < 0 or end > len(self):
            raise ValueError(
                f"Range {start}:{end} is out of bounds for index of size {len(self)}"
            )

        if start >= end:
            raise ValueError(f"Start {start} must be less than end {end}")

        return SimpleIndex(self.__index[start:end])

    def intersection(self, other: DataIndex) -> DataIndex:
        if len(self) == 0 or len(other) == 0:
            return SimpleIndex.empty()
        other_mask = other.into_mask()
        self_mask = self.into_mask()
        length = max(len(other_mask), len(self_mask))
        self_mask.resize(length)
        other_mask.resize(length)
        new_idx = np.where(self_mask & other_mask)[0]
        return SimpleIndex(new_idx)

    def mask(self, mask: np.ndarray) -> DataIndex:
        if mask.shape != self.__index.shape:
            raise np.exceptions.AxisError(
                f"Mask shape {mask.shape} does not match index size {len(self)}"
            )

        if mask.dtype != bool:
            raise TypeError(f"Mask dtype {mask.dtype} is not boolean")

        if not mask.any():
            return SimpleIndex.empty()

        if mask.all():
            return self

        return SimpleIndex(self.__index[mask])

    def get_data(self, data: h5py.Dataset) -> np.ndarray:
        """
        Get the data from the dataset using the index.
        """
        if not isinstance(data, (h5py.Dataset, np.ndarray)):
            raise ValueError("Data must be a h5py.Dataset")
        if len(self) == 0:
            return np.array([], dtype=data.dtype)

        min_index = self.__index.min()
        max_index = self.__index.max()
        output = data[min_index : max_index + 1]
        indices_into_output = self.__index - min_index
        return output[indices_into_output]

    def __getitem__(self, item: int) -> DataIndex:
        """
        Get an item from the index.
        """
        if item < 0 or item >= len(self):
            raise IndexError(
                f"Index {item} out of bounds for index of size {len(self)}"
            )
        val = self.__index[item]
        return SimpleIndex(np.array([val]))


def all_are_simple(others: tuple[DataIndex, ...]) -> TypeGuard[tuple[SimpleIndex, ...]]:
    """
    Check if all elements in the tuple are instances of SimpleIndex.
    """
    return all(isinstance(other, SimpleIndex) for other in others) or not others
