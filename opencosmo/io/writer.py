from typing import Protocol, Iterable, Optional
from pathlib import Path
from opencosmo.dataset.index import DataIndex

class DataWriter(Protocol):

    def write(self, file: Path, index: DataIndex, columns: Iterable[str], dataset_name: Optional[str]): ...


