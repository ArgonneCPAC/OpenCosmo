from typing import Protocol, Iterable, Optional, Self
from pathlib import Path
from opencosmo.dataset.index import DataIndex
import h5py

class DataWriter(Protocol):


    @classmethod
    def prepare(cls, file: Path | h5py.File): ...

    def write_dataset(self, data: h5py.Group, index: DataIndex, columns: Iterable[str], dataset_path: Optional[str]): ...

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

    def write_data(self, data: h5py.Group, index: DataIndex, columns: Iterable[str], dataset_path: Optional[str]:


    




