from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
from h5py import File

import opencosmo as oc
from opencosmo import link as l


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
    def open(
        cls, file: File, datasets_to_get: Optional[Iterable[str]] = None
    ) -> LinkedCollection:
        return l.open_linked_file(file, datasets_to_get)

    @classmethod
    def read(cls, *args, **kwargs) -> LinkedCollection:
        raise NotImplementedError

    @property
    def properties(self) -> oc.Dataset:
        """
        Return the properties dataset.
        """
        return self.__properties

    def keys(self) -> list[str]:
        """
        Return the keys of the linked datasets.
        """
        return list(self.__handlers.keys()) + [self.__properties.header.file.data_type]

    def values(self) -> list[oc.Dataset]:
        """
        Return the linked datasets.
        """
        return [self.__properties] + [
            handler.get_all_data() for handler in self.__handlers.values()
        ]

    def items(self) -> list[tuple[str, oc.Dataset]]:
        """
        Return the linked datasets as key-value pairs.
        """
        return [
            (key, handler.get_all_data()) for key, handler in self.__handlers.items()
        ]

    def __getitem__(self, key: str) -> oc.Dataset:
        """
        Return the linked dataset with the given key.
        """
        if key == self.__properties.header.file.data_type:
            return self.__properties
        elif key not in self.__handlers:
            raise KeyError(f"Dataset {key} not found in collection.")
        return self.__handlers[key].get_all_data()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for dataset in self.values():
            try:
                dataset.__exit__(*args)
            except AttributeError:
                continue

    def select(self, dataset: str, columns: str | list[str]) -> LinkedCollection:
        """
        Update the linked collection to only include the columns specified
        in the given dataset.
        """
        if dataset == self.__properties.header.file.data_type:
            new_properties = self.__properties.select(columns)
            return LinkedCollection(
                new_properties,
                self.__handlers,
            )

        elif dataset not in self.__handlers:
            raise ValueError(f"Dataset {dataset} not found in collection.")
        handler = self.__handlers[dataset]
        new_handler = handler.select(columns)
        return LinkedCollection(
            self.__properties, {**self.__handlers, dataset: new_handler}
        )

    def filter(self, *masks):
        """
        Apply a filter to the properties dataset and propagate it to the linked datasets
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

    def objects(
        self, data_types: Optional[Iterable[str]] = None
    ) -> Iterable[tuple[dict[str, Any], dict[str, Optional[oc.Dataset]]]]:
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
            index = np.array(self.__properties.indices[i])
            output = {key: handler.get_data(index) for key, handler in handlers.items()}
            if not any(output.values()):
                continue
            yield row, output

    def write(self, file: File):
        header = self.__properties.header
        header.write(file)
        self.__properties.write(file, header.file.data_type)
        link_group = file[header.file.data_type].create_group("data_linked")
        keys = list(self.__handlers.keys())
        keys.sort()
        for key in keys:
            handler = self.__handlers[key]
            handler.write(file, link_group, key, self.__idxs)
