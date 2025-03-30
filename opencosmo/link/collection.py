from __future__ import annotations
from typing import Optional, Iterable
import opencosmo as oc
from opencosmo import link as l
import numpy as np

class LinkedCollection:
    """
    A collection of datasets that are linked together, allowing
    for cross-matching and other operations to be performed.

    For now, these are always a combination of a properties dataset
    and several particle or profile datasets. 
    """

    def __init__(
        self,
        properties: oc.Dataset,
        handlers: dict[str, l.LinkHandler],
        *args,
        **kwargs,
    ):
        """
        Initialize a linked collection with the provided datasets and links.
        """

        self.__properties = properties
        self.__handlers = handlers
        self.__idxs = self.__properties.indices

    @classmethod
    def open(cls, file: File, names: Optional[Iterable[str]] = None) -> LinkedCollection:
        return l.open_linked_file(file)

    @classmethod
    def read(cls, *args, **kwargs) -> LinkedCollection:
        raise NotImplementedError



    def as_dict(self) -> dict[str, oc.Dataset]:
        return self

    def keys(self) -> list[str]:
        """
        Return the keys of the linked datasets.
        """
        return list(self.__handlers.keys()) + [self.__properties.header.file.data_type]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for dataset in self.values():
            try:
                dataset.__exit__(*args)
            except AttributeError:
                continue

    def filter(self, *masks):
        """
        Apply a filter to the properties dataset and propagate it to the linked datasets.
        """
        if not masks:
            return self
        filtered = self.__properties.filter(*masks)
        return LinkedCollection(
            filtered,
            self.__handlers,
        )
    def take(self, n: int, at: str = "start"):
        new_properties = self.__properties.take(n, at)
        return LinkedCollection(
            new_properties,
            self.__handlers,
        )


    def objects(self, data_types: Optional[Iterable[str]] = None) -> Iterable[tuple[oc.Dataset, dict[str, oc.Dataset]]]:
        """
        Iterate over the properties dataset and the linked datasets.
        """
        if data_types is None:
            handlers = self.__handlers
        elif not all(dt in self.__handlers for dt in data_types):
            raise ValueError("Some data types are not linked in the collection.")
        else:
            handlers = {dt: self.__handlers[dt] for dt in data_types}

        for i, row in enumerate(self.__properties.rows()):
            index = np.array([i])
            output = {key: handler.get_data(index) for key, handler in handlers.items()}
            if not any(output.values()):
                continue
            yield row, output

    def write(self, file: File):
        header = self.__properties.header
        header.write(file)
        self.__properties.write(file, header.file.data_type)
        link_group = file[header.file.data_type].create_group("data_linked")
        for key, handler in self.__handlers.items():
            handler.write(file, key, link_group, self.__idxs)

 
