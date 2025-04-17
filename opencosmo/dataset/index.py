from __future__ import annotations

from typing import Any, Protocol, TypeGuard, TypeVar

import h5py
import numpy as np

T = TypeVar("T", np.ndarray, h5py.Dataset)


class EmptyMaskError(Exception):
    pass


def all_are_chunked(
    others: tuple[DataIndex, ...],
) -> TypeGuard[tuple[ChunkedIndex, ...]]:
    """
    Check if all elements in the tuple are instances of ChunkedIndex.
    """
    return all(isinstance(other, ChunkedIndex) for other in others)


def all_are_simple(others: tuple[DataIndex, ...]) -> TypeGuard[tuple[SimpleIndex, ...]]:
    """
    Check if all elements in the tuple are instances of SimpleIndex.
    """
    return all(isinstance(other, SimpleIndex) for other in others)


class DataIndex(Protocol):
    @classmethod
    def from_size(cls, size: int) -> DataIndex: ...
    def set_data(self, data: np.ndarray, value: Any) -> np.ndarray: ...
    def get_data(self, data: h5py.Dataset | np.ndarray) -> np.ndarray: ...
    def take(self, n: int, at: str = "random") -> DataIndex: ...
    def take_range(self, start: int, end: int) -> DataIndex: ...
    def mask(self, mask: np.ndarray) -> DataIndex: ...
    def range(self) -> tuple[int, int]: ...
    def concatenate(self, *others: DataIndex) -> DataIndex: ...
    def __len__(self) -> int: ...
    def __getitem__(self, item: int) -> DataIndex: ...


class SimpleIndex:
    """
    An index of integers.
    """

    def __init__(self, index: np.ndarray) -> None:
        self.__index = np.sort(index)

    @classmethod
    def from_size(cls, size: int) -> DataIndex:
        return SimpleIndex(np.arange(size))

    def __len__(self) -> int:
        return len(self.__index)

    def range(self) -> tuple[int, int]:
        """
        Guranteed to be sorted
        """
        return self.__index[0], self.__index[-1]

    def concatenate(self, *others: DataIndex) -> DataIndex:
        if len(others) == 0:
            return self
        if all_are_simple(others):
            new_index = np.concatenate(
                [self.__index] + [other.__index for other in others]
            )
            new_index = np.sort(np.unique(new_index))
            return SimpleIndex(new_index)
        else:
            simple_indices = map(
                lambda x: x.to_simple_index() if isinstance(x, ChunkedIndex) else x,
                others,
            )
            return self.concatenate(*simple_indices)

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

    def mask(self, mask: np.ndarray) -> DataIndex:
        if mask.shape != self.__index.shape:
            raise np.exceptions.AxisError(
                f"Mask shape {mask.shape} does not match index size {len(self)}"
            )

        if mask.dtype != bool:
            raise TypeError(f"Mask dtype {mask.dtype} is not boolean")

        if not mask.any():
            raise EmptyMaskError("Mask is all False")

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


def pack(start: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine adjacent chunks into a single chunk.
    """

    # Calculate the end of each chunk
    end = start + size

    # Determine where a new chunk should start (i.e., not adjacent to previous)
    # We prepend True for the first chunk to always start a group
    new_group = np.ones(len(start), dtype=bool)
    new_group[1:] = start[1:] != end[:-1]

    # Assign a group ID for each segment
    group_ids = np.cumsum(new_group)

    # Combine chunks by group
    combined_start = np.zeros(group_ids[-1], dtype=start.dtype)
    combined_size = np.zeros_like(combined_start)

    np.add.at(combined_start, group_ids - 1, np.where(new_group, start, 0))
    np.add.at(combined_size, group_ids - 1, size)

    return combined_start, combined_size


class ChunkedIndex:
    def __init__(self, starts: np.ndarray, sizes: np.ndarray) -> None:
        # sort the starts and sizes
        # pack the starts and sizes
        self.__starts = starts
        self.__sizes = sizes

    def range(self) -> tuple[int, int]:
        """
        Get the range of the index.
        """
        return self.__starts[0], self.__starts[-1] + self.__sizes[-1] - 1

    def to_simple_index(self) -> SimpleIndex:
        """
        Convert the ChunkedIndex to a SimpleIndex.
        """
        idxs = np.concatenate(
            [
                np.arange(start, start + size)
                for start, size in zip(self.__starts, self.__sizes)
            ]
        )
        idxs = np.unique(idxs)
        return SimpleIndex(idxs)

    def concatenate(self, *others: DataIndex) -> DataIndex:
        if len(others) == 0:
            return self
        if all_are_chunked(others):
            new_starts = np.concatenate(
                [self.__starts] + [other.__starts for other in others]
            )
            new_sizes = np.concatenate(
                [self.__sizes] + [other.__sizes for other in others]
            )
            return ChunkedIndex(new_starts, new_sizes)

        else:
            simple_indices = map(
                lambda x: x.to_simple_index() if isinstance(x, ChunkedIndex) else x,
                others,
            )
            return self.concatenate(*simple_indices)

    @classmethod
    def from_size(cls, size: int) -> DataIndex:
        """
        Create a ChunkedIndex from a size.
        """
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")
        # Create an array of chunk sizes

        starts = np.array([0])
        sizes = np.array([size])
        return ChunkedIndex(starts, sizes)

    @classmethod
    def single_chunk(cls, start: int, size: int) -> ChunkedIndex:
        """
        Create a ChunkedIndex with a single chunk.
        """
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")
        if start < 0:
            raise ValueError(f"Start must be non-negative, got {start}")
        starts = np.array([start])
        sizes = np.array([size])
        return ChunkedIndex(starts, sizes)

    def set_data(self, data: np.ndarray, value: bool) -> np.ndarray:
        """
        Set the data at the index to the given value.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")

        for start, size in zip(self.__starts, self.__sizes):
            data[start : start + size] = value
        return data

    def __len__(self) -> int:
        """
        Get the total size of the index.
        """
        return np.sum(self.__sizes)

    def take(self, n: int, at: str = "random") -> DataIndex:
        if n > len(self):
            raise ValueError(f"Cannot take {n} elements from index of size {len(self)}")

        if at == "random":
            idxs = np.concatenate(
                [
                    np.arange(start, start + size)
                    for start, size in zip(self.__starts, self.__sizes)
                ]
            )
            idxs = np.random.choice(idxs, n, replace=False)
            return SimpleIndex(idxs)

        elif at == "start":
            last_chunk_in_range = np.searchsorted(np.cumsum(self.__sizes), n)
            new_starts = self.__starts[: last_chunk_in_range + 1].copy()
            new_sizes = self.__sizes[: last_chunk_in_range + 1].copy()
            new_sizes[-1] = n - np.sum(new_sizes[:-1])
            return ChunkedIndex(new_starts, new_sizes)

        elif at == "end":
            starting_chunk = np.searchsorted(np.cumsum(self.__sizes), len(self) - n)
            new_sizes = self.__sizes[starting_chunk:].copy()
            new_starts = self.__starts[starting_chunk:].copy()
            new_sizes[0] = n - np.sum(new_sizes[1:])
            new_starts[0] = (
                self.__starts[starting_chunk]
                + self.__sizes[starting_chunk]
                - new_sizes[0]
            )
            return ChunkedIndex(new_starts, new_sizes)
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

        # Get the indices of the chunks that are in the range
        idxs = np.concatenate(
            [
                np.arange(start, start + size)
                for start, size in zip(self.__starts, self.__sizes)
            ]
        )
        range_idxs = idxs[start:end]

        return SimpleIndex(range_idxs)

    def mask(self, mask: np.ndarray) -> DataIndex:
        """
        Mask the index with a boolean mask.
        """
        if mask.shape != (len(self),):
            raise ValueError(
                f"Mask shape {mask.shape} does not match index size {len(self)}"
            )

        if mask.dtype != bool:
            raise ValueError(f"Mask dtype {mask.dtype} is not boolean")

        if not mask.any():
            raise EmptyMaskError("Mask is all False")

        if mask.all():
            return self

        # Get the indices of the chunks that are masked
        idxs = np.concatenate(
            [
                np.arange(start, start + size)
                for start, size in zip(self.__starts, self.__sizes)
            ]
        )
        masked_idxs = idxs[mask]

        return SimpleIndex(masked_idxs)

    def get_data(self, data: h5py.Dataset | np.ndarray) -> np.ndarray:
        """
        Get the data from the dataset using the index. We want to perform as few reads
        as possible. However, the chunks may not be continuous. This method sorts the
        chunks so it can read the data in the largest possible chunks, it then
        reshuffles the data to match the original order.

        For large numbers of chunks, this is much much faster than reading each chunk
        in the order they are stored in the index. I know because I tried. It sucked.
        """
        if not isinstance(data, (h5py.Dataset, np.ndarray)):
            raise ValueError("Data must be a h5py.Dataset")

        if len(self) == 0:
            return np.array([], dtype=data.dtype)
        if len(self.__starts) == 1:
            return data[self.__starts[0] : self.__starts[0] + self.__sizes[0]]

        sorted_start_index = np.argsort(self.__starts)
        new_starts = self.__starts[sorted_start_index]
        new_sizes = self.__sizes[sorted_start_index]

        packed_starts, packed_sizes = pack(new_starts, new_sizes)

        shape = (len(self),) + data.shape[1:]
        temp = np.zeros(shape, dtype=data.dtype)
        running_index = 0
        for i, (start, size) in enumerate(zip(packed_starts, packed_sizes)):
            temp[running_index : running_index + size] = data[start : start + size]
            running_index += size

        output = np.zeros(len(self), dtype=data.dtype)
        cumulative_sorted_sizes = np.insert(np.cumsum(new_sizes), 0, 0)
        cumulative_original_sizes = np.insert(np.cumsum(self.__sizes), 0, 0)

        # reshuffle the output to match the original order
        for i, sorted_index in enumerate(sorted_start_index):
            start = cumulative_original_sizes[sorted_index]
            end = cumulative_original_sizes[sorted_index + 1]
            data = temp[cumulative_sorted_sizes[i] : cumulative_sorted_sizes[i + 1]]
            output[start:end] = data

        return output

    def __getitem__(self, item: int) -> DataIndex:
        """
        Get an item from the index.
        """
        if item < 0 or item >= len(self):
            raise IndexError(
                f"Index {item} out of bounds for index of size {len(self)}"
            )
        sums = np.cumsum(self.__sizes)
        index = np.searchsorted(sums, item)
        start = self.__starts[index]
        offset = item - sums[index - 1] if index > 0 else item
        return SimpleIndex(np.array([start + offset]))
