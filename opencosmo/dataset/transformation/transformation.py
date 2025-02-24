from typing import Optional, Protocol

from astropy.table import Column


class ColumnTransformation(Protocol):
    def __call__(self, column: Column) -> Optional[Column]: ...
