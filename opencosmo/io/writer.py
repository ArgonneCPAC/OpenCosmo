from pathlib import Path
from typing import Iterable, Optional, Protocol, Self

import h5py

from opencosmo.dataset.index import DataIndex
from opencosmo.io.schema import ColumnSchema, DatasetSchema, IndexSchema, LinkSchema


def generate_dataset_schema(
    input_dataset_group: h5py.Group,
    column_names: Iterable[str],
    n_elements: int,
    link_schemas: list[LinkSchema] = [],
    index_schemas: list[IndexSchema] = [],
    additional_columns: list[ColumnSchema] = [],
    include_header: bool = True,
) -> DatasetSchema:
    column_schemas = []
    input_data_group = input_dataset_group["data"]
    input_column_names = set(input_data_group.keys())
    if not set(column_names).issubset(input_column_names):
        raise ValueError(
            "Dataset schema recieved columns that are not in the original dataset!"
        )

    for name in column_names:
        col = input_data_group[name]
        shape = col.shape
        new_shape = (n_elements,) + shape[1:]
        column_schemas.append(ColumnSchema(name, new_shape, col.dtype))

    column_schemas.extend(additional_columns)
    return DatasetSchema(column_schemas, link_schemas, index_schemas, include_header)


class DataWriter(Protocol):
    @classmethod
    def prepare(cls, file: Path | h5py.File): ...

    def write_dataset(
        self,
        data: h5py.Group,
        index: DataIndex,
        columns: Iterable[str],
        dataset_path: Optional[str],
    ): ...

    def allocate(): ...

    def write_tree(): ...

    def write_header(): ...


class SerialWriter:
    def __init__(self, file: h5py.File):
        self.__file = h5py.File

    @classmethod
    def prepare(cls, file: Path | h5py.File) -> Self:
        if isinstance(file, Path):
            if file.exists():
                raise FileExistsError(file)
            file = h5py.File(file, "w")
        return SerialWriter(file)
