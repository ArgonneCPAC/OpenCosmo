from functools import partial, singledispatch
from typing import Any, Callable, Optional, TypeVar

import h5py
import numpy as np

from opencosmo.index import ChunkedIndex, DataIndex

T = TypeVar("T", h5py.Dataset, h5py.Group)


Updater = Callable[[np.ndarray], np.ndarray]


class IndexMap:
    """
    An IndexMap provides a mapping between a selection of elements (possibly
    non-contiguous) in some input hdf5 dataset to a contiguous section in some output
    hdf5 dataset. It is used primarily when chunks of data must be interwoven into a
    single dataset when they are written, rather than being written sequentially.

    The IndexMap assumes that both datasets already exist. It is not responsible for
    allocating space in the output.

    It can also perform this operation with groups. If it is a group, the assumption
    is that all datasets that exist in the input also exist with the output.
    """

    def __init__(self, in_: DataIndex, out: ChunkedIndex):
        if len(in_) != len(out):
            raise ValueError("IndexMap recieved indices that were not the same length!")
        self.__input = in_
        self.__output = out

    @property
    def input(self):
        return self.__input

    @property
    def output(self):
        return self.__output

    @classmethod
    def one_to_one(cls, size: int):
        in_ = ChunkedIndex.from_size(size)
        out = ChunkedIndex.from_size(size)
        return IndexMap(in_, out)

    def get_data(self, dataset: h5py.Dataset):
        return self.input.get_data(dataset)

    def __len__(self):
        return len(self.__input)

    def transfer(self, data_in: T, data_out: T, updater: Optional[Updater]) -> T:
        return transfer_data(data_in, self.__input, data_out, self.__output, updater)


@singledispatch
def transfer_data(data_in: Any, *args, **kwargs):
    raise ValueError(f"Invalid type for input dataset {type(data_in)}")


@transfer_data.register
def _(
    data_in: h5py.Dataset,
    index_in: DataIndex,
    data_out: h5py.Dataset,
    index_out: ChunkedIndex,
    updater: Optional[Updater],
):
    data = index_in.get_data(data_in)
    if updater is not None:
        data = updater(data)

    index_out.write_dataset(data, data_out)


@transfer_data.register
def _(
    data_in: h5py.Group,
    index_in: DataIndex,
    data_out: h5py.Group,
    index_out: ChunkedIndex,
    updater: Optional[Updater],
):
    data_in_names = set(data_in.keys())
    data_out_names = set(data_out.keys())
    if not data_in_names.issubset(data_out_names):
        raise ValueError(
            "Output group must have at least all columns from the input group!"
        )

    visitor_ = partial(
        visitor,
        index_in=index_in,
        data_out=data_out,
        index_out=index_out,
        updater=updater,
    )
    data_in.visititems(visitor_)


def visitor(
    name: str,
    data_in: h5py.Dataset,
    index_in: DataIndex,
    data_out: h5py.Group,
    index_out: ChunkedIndex,
    updater: Optional[Updater],
):
    dataset_out = data_out.get(name)
    return transfer_data(data_in, index_in, dataset_out, data_out, index_out, updater)
