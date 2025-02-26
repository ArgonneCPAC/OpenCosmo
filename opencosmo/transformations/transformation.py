from typing import Optional, Protocol

from astropy.table import Column, Table
from h5py import Dataset
from numpy.typing import NDArray

TransformationOutput = Column | Table | NDArray[bool]


class Transformation(Protocol):
    """
    A transformation that can be applied to a hdf5 dataset, or an astropy table/column.
    output should be an astropy table or column, or a numpy array of booleans (mask).

    If the transformation cannot be applied to the data, it should
    return None
    """

    def __call__(self, input: Column | Table) -> Optional[TransformationOutput]: ...


class DatasetTransformation(Transformation):
    """
    A transformation that can be applied to a dataset. In OpenCosmo, all
    datasets are single-column.
    """

    def __call__(self, input: Dataset) -> Optional[Column]: ...


class TableTransformation(Transformation):
    """
    A transformation that can be applied to a table, producing a new table

    Columns with the same names in both will be updated, and new columns
    will be added.
    """

    def __call__(self, input: Table) -> Optional[Table]: ...


class ColumnTransformation(Transformation):
    """
    A transformation that is applied to a single column, producing
    an updated version of that column (which will replace the original)
    """

    def __init__(self, column_name: str): ...

    @property
    def column_name(self) -> str: ...

    def __call__(self, input: Column) -> Optional[Column]: ...


class FilterTransformation(Transformation):
    """
    A transformation that masks rows of a table based on some criteria.
    The mask should be a boolean array with the same length as the table.
    """

    def __call__(self, input: Table) -> Optional[NDArray[bool]]: ...
