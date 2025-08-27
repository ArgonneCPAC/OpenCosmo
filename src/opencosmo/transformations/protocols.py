from enum import Enum
from typing import Optional, Protocol, Sequence, runtime_checkable

import h5py
from astropy.table import Table  # type: ignore
from numpy.typing import ArrayLike


class TransformationType(Enum):
    TABLE = "table"
    COLUMN = "column"
    ALL_COLUMNS = "all_columns"
    FILTER = "filter"


class TransformationGenerator(Protocol):
    """
    A transformation generator is a callable that returns a transformation
    or set of transformations based on information stored in the attributes of a given
    dataset. Examples include units stored as attributes
    """

    def __call__(self, column: h5py.Dataset) -> "TransformationDict": ...


class TableTransformation(Protocol):
    """
    A transformation that can be applied to a table, producing a new table.

    The new table will replace the original table.
    """

    def __call__(self, input: Table) -> Table: ...


@runtime_checkable
class AllColumnTransformation(Protocol):
    """
    A transformation that is applied to all columns in a table.
    """

    def __call__(self, input: ArrayLike) -> ArrayLike: ...


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

    def __call__(self, input: ArrayLike) -> ArrayLike: ...


Transformation = TableTransformation | ColumnTransformation | AllColumnTransformation
TransformationDict = dict[TransformationType, Sequence[Transformation]]
