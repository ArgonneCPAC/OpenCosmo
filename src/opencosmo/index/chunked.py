from typing import TypeGuard

import h5py
import numpy as np
from numpy.typing import NDArray

from opencosmo.index import simple
from opencosmo.index.protocols import DataIndex


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

    def into_array(self) -> NDArray[np.int_]:
        """
        Convert the ChunkedIndex to a SimpleIndex.
        """
        if len(self) == 0:
            return simple.SimpleIndex.empty()
        idxs = np.concatenate(
            [
                np.arange(start, start + size)
                for start, size in zip(self.__starts, self.__sizes)
            ]
        )
        return np.unique(idxs)

    def into_mask(self) -> np.ndarray:
        ends = self.__starts + self.__sizes
        mask = np.zeros(np.max(ends), dtype=bool)
        for start, end in np.nditer([self.__starts, ends]):
            mask[start:end] = True
        return mask

    def intersection(self, other: DataIndex) -> DataIndex:
        if len(self) == 0 or len(other) == 0:
            return simple.SimpleIndex.empty()
        other_mask = other.into_mask()
        self_mask = self.into_mask()
        length = max(len(other_mask), len(self_mask))
        self_mask.resize(length)
        other_mask.resize(length)
        new_idx = np.where(self_mask & other_mask)[0]
        return simple.SimpleIndex(new_idx)

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
            indexes = np.concatenate([o.into_array() for o in others])
            indexes = np.concatenate(self.into_array(), indexes)
            return simple.SimpleIndex(indexes)

    @classmethod
    def from_size(cls, size: int) -> "ChunkedIndex":
        """
        Create a ChunkedIndex from a size.
        """
        if size <= 0:
            return ChunkedIndex.empty()
        # Create an array of chunk sizes

        starts = np.array([0])
        sizes = np.array([size])
        return ChunkedIndex(starts, sizes)

    @classmethod
    def single_chunk(cls, start: int, size: int) -> "ChunkedIndex":
        """
        Create a ChunkedIndex with a single chunk.
        """
        if start < 0:
            raise ValueError(f"Start must be non-negative, got {start}")

        if not size:
            return ChunkedIndex.empty()
        starts = np.array([start])
        sizes = np.array([size])
        return ChunkedIndex(starts, sizes)

    @classmethod
    def empty(cls):
        start = np.array([], dtype=int)
        size = np.array([], dtype=int)
        return ChunkedIndex(start, size)

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
            return simple.SimpleIndex(idxs)

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

        if start > end:
            raise ValueError(f"Start {start} must be less than end {end}")

        if start == end:
            return ChunkedIndex.empty()

        # Get the indices of the chunks that are in the range
        idxs = np.concatenate(
            [
                np.arange(start, start + size)
                for start, size in zip(self.__starts, self.__sizes)
            ]
        )
        range_idxs = idxs[start:end]

        return simple.SimpleIndex(range_idxs)

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
            return simple.SimpleIndex.empty()

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

        return simple.SimpleIndex(masked_idxs)

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

    def n_in_range(
        self, start: NDArray[np.int_], size: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        """
        Return the number of elements in this index that fall within
        a specified data range. Used to mask spatial index.


        As with numpy, this is the half-open range [start, end)
        """
        if len(start) != len(size):
            raise ValueError("Start and size arrays must have the same length")
        if np.any(size < 0):
            raise ValueError("Sizes must greater than or equal to zero")
        if len(self) == 0:
            return np.zeros_like(start)

        required_memory = len(start) * len(self.__starts) * 8
        MEMORY_LIMIT = 2_000_000_000
        if required_memory > MEMORY_LIMIT:
            chunk_size = MEMORY_LIMIT // (8 * len(self.__starts))
            n_chunks = np.ceil(len(start) / chunk_size)
        else:
            n_chunks = 1

        ends = start + size
        self_ends = self.__starts + self.__sizes

        rs = 0
        output = np.zeros_like(start)
        for arr in np.array_split(start, n_chunks):
            in_range_starts = np.clip(
                self.__starts[:, np.newaxis],
                a_min=arr,
                a_max=ends[rs : rs + len(arr)],
            )
            in_range_ends = np.clip(
                self_ends[:, np.newaxis],
                a_min=arr,
                a_max=ends[rs : rs + len(arr)],
            )
            output[rs : rs + len(arr)] = np.sum(in_range_ends - in_range_starts, axis=0)
            rs += len(arr)

        return output
        # example: row[0] in in_range_starts tells me what happens if I clip the 0th
        # element in self.__starts to each of the values in the start and end array
        # that have been provided in the call.

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
        return simple.SimpleIndex(np.array([start + offset]))


def all_are_chunked(
    others: tuple[DataIndex, ...],
) -> TypeGuard[tuple[ChunkedIndex, ...]]:
    """
    Check if all elements in the tuple are instances of ChunkedIndex.
    """
    return all(isinstance(other, ChunkedIndex) for other in others) or not others
