from typing import Iterable, Optional
from numpy.typing import NDArray, DTypeLike
import numpy as np
from h5py import Group

ColumnShape = tuple[int, ...]
LinkIndex = NDArray[int]


def validate_shapes(columns: list["ColumnSchema"], links: Optional[Iterable["LinkSchema"]] = None):
    """
    Datasets should be representable as tables, so the first dimension
    of all shapes must be the same.
    """
    n_rows = set(c.shape[0] for c in columns)
    if len(n_rows) > 1:
        raise ValueError("All columns in a dataset must have the same length!")

    if links is not None:
        validate_link_schemas(links)

def validate_link_schemas(n_cols: int, links: Iterable["LinkSchema"]):
    if any(link.n > n_cols for link in links):
        raise ValueError("All links must have the same length as the dataset!")



def verify_file_schema(datasets: Iterable["DatasetSchema"], has_header: bool = False):
    all_datasets_have_header = all(ds.has_header for ds in datasets)
    if not has_header and not all_datasets_have_header:
        raise ValueError("Either the top-level file must have a header, or all datsets must")

class FileSchema: 
    def __init__(self, datasets: dict[str, "DatasetSchema"], has_header: bool = False):
        self.datasets = list(datasets)
        self.has_header = has_header
        verify_file_schema(self.datasets.values(), self.has_header)

class LinkSchema:
    
    def __init__(self, name: str, n: int, has_sizes: bool = False):
        self.n = n
        self.has_sizes = has_sizes
        self.name = name

    def allocate(self, group: Group):
        if self.has_sizes:
            start_name = f"{self.name}_start"
            size_name = f"{self.name}_size"
            group.create_dataset(start_name, shape=(self.n,), dtype=np.uint64)
            group.create_dataset(size_name, shape=(self.n,), dtype=np.uint64)
        else:
            name = f"{self.name}_idx"
            group.create_dataset(name, shape=(self.n,), dtype=np.uint64)

class IndexSchema:
    pass


class ColumnSchema:
    def __init__(self, name: str, shape: ColumnShape, dtype: DTypeLike):
        self.name == name
        self.shape == shape
        self.dtype = dtype

    def allocate(self, group: Group):
        group.require_dataset(self.name, self.shape, self.dtype)

class DatasetSchema:
    def __init__(self, columns: Iterable[ColumnSchema], links: Optional[Iterable[LinkSchema]] = None, has_header: bool = False):
        self.columns = list(columns)
        self.has_header = has_header
        self.links = links
        if len(set(c.name for c in self.columns)) != len(self.columns):
            raise ValueError("Column names must be unique!")
        validate_shapes(self.columns, self.links)

    def allocate(self, group: Group):
        data_group = group.require_group("data")
        for column in self.columns:
            column.allocate(data_group)






