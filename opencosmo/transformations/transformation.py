from __future__ import annotations

from enum import Enum
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from astropy.table import Column, Table  # type: ignore
from numpy.typing import NDArray


class TransformationType(Enum):
    TABLE = "table"
    COLUMN = "column"
    ALL_COLUMNS = "all_columns"
    FILTER = "filter"


class TableTransformation(Protocol):
    """
    A transformation that can be applied to a table, producing a new table.

    The new table will replace the original table.
    """

    def __call__(self, input: Table) -> Optional[Table]: ...


@runtime_checkable
class AllColumnTransformation(Protocol):
    """
    A transformation that is applied to all columns in a table.
    """

    def __call__(self, input: Column) -> Optional[Column]: ...


class ColumnTransformation(Protocol):
    """
    A transformation that is applied to a single column, producing
    an updated version of that version of that column.

    An "all_columns" transformation is just a regular column transformation
    except that it will be applied to all columns in the table. In this case,
    column_name should return None.
    """

    def __init__(self, column_name: str, *args, **kwargs): ...

    @property
    def column_name(self) -> Optional[str]: ...

    def __call__(self, input: Column) -> Optional[Column]: ...


class FilterTransformation(Protocol):
    """
    A transformation that masks rows of a table based on some criteria.
    The mask should be a boolean array with the same length as the table.
    """

    def __call__(self, input: Table) -> Optional[NDArray[np.bool_]]: ...


Transformation = (
    TableTransformation
    | ColumnTransformation
    | FilterTransformation
    | AllColumnTransformation
)
TransformationDict = dict[TransformationType, list[Transformation]]
