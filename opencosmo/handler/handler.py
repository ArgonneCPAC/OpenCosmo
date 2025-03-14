from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

import h5py
import numpy as np
from astropy.table import Column, Table  # type: ignore

from opencosmo.dataset.column import ColumnBuilder


class OpenCosmoDataHandler(Protocol):
    """
    There are multiple modes we can imagine working with data in. For
    small data, it's totally fine (and probably preferable) to just load
    the dataset into memory immediately. For large data, we want to keep
    a handle to the file and only load requested data when it's needed.

    This class defines a protocol that handlers must implement. These
    handlers will be used by the Dataset class to handle data. Because the
    handler is separate from the dataset, we can have multiple datasets
    that use a single handler.

    The handler has a few responsibilities:

    1. It should only require a path to the data to work
    2. It should be a context manager
    3. It needs to be able to apply masks and transformations to the datahandler
    4. It should be able to return the data

    The handler is only responsible for working with the actual data. Indexes
    and metadata are handled separately.
    """

    def __init__(self, file: Path | h5py.File | dict): ...
    def __enter__(self): ...
    def __exit__(self, *exc_details): ...
    def __len__(self) -> int: ...
    def collect(
        self, columns: Iterable[str], mask: np.ndarray
    ) -> OpenCosmoDataHandler: ...
    def write(
        self,
        file: h5py.File,
        mask: np.ndarray,
        columns: Iterable[str],
        dataset_name="data",
    ) -> None: ...
    def get_data(
        self, column_builders: dict[str, ColumnBuilder], mask: np.ndarray
    ) -> Column | Table: ...

    def take_mask(self, n: int, strategy: str, mask: np.ndarray) -> np.ndarray: ...
