from __future__ import annotations

from typing import Optional, Protocol

import numpy as np
from astropy.table import Column, Table  # type: ignore
from numpy.typing import NDArray


class TableTransformation(Protocol):
    """
    A transformation that can be applied to a table, producing a new table.

    The new table will replace the original table.
    """

    def __call__(self, input: Table) -> Optional[Table]: ...


class ColumnTransformation(Protocol):
    """
    A transformation that is applied to a single column, producing
    an updated version of that version of that column.
    """

    def __init__(self, column_name: str, *args, **kwargs): ...

    @property
    def column_name(self) -> str: ...

    def __call__(self, input: Column) -> Optional[Column]: ...


class FilterTransformation(Protocol):
    """
    A transformation that masks rows of a table based on some criteria.
    The mask should be a boolean array with the same length as the table.
    """

    def __call__(self, input: Table) -> Optional[NDArray[np.bool_]]: ...


Transformation = TableTransformation | ColumnTransformation | FilterTransformation
