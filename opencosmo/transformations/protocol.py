from typing import Optional, Protocol

from astropy.table import Column, Table
from numpy.typing import NDArray

TransformationOutput = Column | Table | NDArray[bool]


class Transformation(Protocol):
    """
    A transformation that can be applied to a column or table,
    producing a new column, table, or boolean mask.

    If the transformation cannot be applied to the data, it should
    return None
    """

    def __call__(self, input: Column | Table) -> Optional[TransformationOutput]: ...


class TableTransformation(Transformation):
    """
    A transformation that can be applied to a table, producing a new table

    Columns with the same names in both will be updated, and new columns
    will be added.
    """

    def __call__(self, input: Table) -> Optional[Table]: ...


class ColumnTransformation(Transformation):
    """
    A transformation that is applied to a single transformation, producing
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
