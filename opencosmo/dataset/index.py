from __future__ import annotations
from typing import Protocol 

import numpy as np
import h5py


class DataIndex(Protocol):

    @classmethod
    def from_size(cls, size: int) -> DataIndex: ...
    def get_data(self, data: h5py.Dataset) -> np.ndarray: ...
    def take(self, n: int, at: str = "random") -> DataIndex: ...
    def mask(self, mask: np.ndarray) -> DataIndex: ...
    def __len__(self) -> int: ...


class SimpleIndex:
    """
    An index of integers. 
    """

    def __init__(self, index: np.ndarray) -> None:
        self.__index = np.sort(index)

    @classmethod
    def from_size(cls, size: int) -> SimpleIndex:
        return SimpleIndex(np.arange(size))

    def __len__(self) -> int:
        return len(self.__index)

    def take(self, n: int, at: str = "random") -> SimpleIndex:
        """
        Take n elements from the index.
        """
        if n > len(self):
            raise ValueError(f"Cannot take {n} elements from index of size {len(self)}")
        if at == "random":
            return SimpleIndex(np.random.choice(self.__index, n, replace=False))
        elif at == "start":
            return SimpleIndex(self.__index[:n])
        elif at == "end":
            return SimpleIndex(self.__index[-n:])
        else:
            raise ValueError(f"Unknown value for 'at': {at}")

    def mask(self, mask: np.ndarray) -> SimpleIndex:
        if mask.shape != self.__index.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match index size {len(self)}")

        if mask.dtype != bool:
            raise ValueError(f"Mask dtype {mask.dtype} is not boolean")

        if not mask.any():
            raise ValueError("Mask is all False")

        if mask.all():
            return self
    
        return SimpleIndex(self.__index[mask])

    def get_data(self, data: h5py.Dataset) -> np.ndarray:
        """
        Get the data from the dataset using the index.
        """
        if not isinstance(data, h5py.Dataset):
            raise ValueError("Data must be a h5py.Dataset")

        min_index = self.__index.min()
        max_index = self.__index.max()
        output = data[min_index:max_index + 1]
        indices_into_output = self.__index - min_index
        return output[indices_into_output]

class ChunkedIndex:

    def __init__(self, starts: np.ndarray, sizes: np.ndarray) -> None:
        self.__starts = starts
        self.__sizes = sizes

    @classmethod
    def from_size(cls, size: int) -> ChunkedIndex:
        """
        Create a ChunkedIndex from a size.
        """
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")
        # Create an array of chunk sizes

        starts = np.array([0])
        sizes = np.array([size])
        return ChunkedIndex(starts, sizes)
    
    def __len__(self) -> int:
        """
        Get the total size of the index.
        """
        return np.sum(self.__sizes)

    def take(self, n: int, at: str = "random") -> DataIndex:
        if n > len(self):
            raise ValueError(f"Cannot take {n} elements from index of size {len(self)}")

        if at == "random":
            idxs = np.concatenate([np.arange(start, start + size) for start, size in zip(self.__starts, self.__sizes)])
            idxs = np.random.choice(idxs, n, replace=False)
            return SimpleIndex(idxs)
        elif at == "start":
            last_chunk_in_range = np.searchsorted(np.sum(self.__sizes), n)
            new_starts = self.__starts[:last_chunk_in_range]
            new_sizes = self.__sizes[:last_chunk_in_range]
            new_sizes[-1] = n - np.sum(new_sizes[:-1])
            return ChunkedIndex(new_starts, new_sizes)
        elif at == "end":
            first_chunk_in_range = np.searchsorted(np.sum(self.__sizes), len(self) - n)
            new_starts = self.__starts[first_chunk_in_range:]
            new_sizes = self.__sizes[first_chunk_in_range:]
            new_sizes[0] = n - np.sum(new_sizes[1:])
            return ChunkedIndex(new_starts, new_sizes)

    def mask(self, mask: np.ndarray) -> DataIndex:
        """
        Mask the index with a boolean mask.
        """
        if mask.shape != (len(self),):
            raise ValueError(f"Mask shape {mask.shape} does not match index size {len(self)}")

        if mask.dtype != bool:
            raise ValueError(f"Mask dtype {mask.dtype} is not boolean")

        if not mask.any():
            raise ValueError("Mask is all False")

        if mask.all():
            return self

        # Get the indices of the chunks that are masked
        idxs = np.concatenate([np.arange(start, start + size) for start, size in zip(self.__starts, self.__sizes)])
        masked_idxs = idxs[mask]

        return SimpleIndex(masked_idxs)
        

    def get_data(self, data: h5py.Dataset) -> np.ndarray:
        """
        Get the data from the dataset using the index.
        """
        if not isinstance(data, h5py.Dataset):
            raise ValueError("Data must be a h5py.Dataset")

        output = np.zeros(len(self), dtype=data.dtype)
        running_index = 0
        for start, size in zip(self.__starts, self.__sizes):
            output[running_index:running_index + size] = data[start:start + size]
            running_index += size
        return output
