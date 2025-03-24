from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional
from pathlib import Path

import h5py

from opencosmo.header import OpenCosmoHeader, read_header
import opencosmo as oc
from .link import verify_links, get_links


def get_collection_type(file: h5py.File) -> Callable[..., DataCollection]:
    """
    Determine the type of a file containing multiple datasets. Currently
    we only support multi_simulation and particle.

    multi_simulation == multiple simulations, same data types
    particle == single simulation, multiple particle species
    """
    datasets = [k for k in file.keys() if k != "header"]
    if len(datasets) == 0:
        raise ValueError("No datasets found in file.")

    if all("particle" in dataset for dataset in datasets) and "header" in file.keys():
        return ParticleCollection

    elif "header" not in file.keys():
        config_values = defaultdict(list)
        for dataset in datasets:
            try:
                filetype_data = dict(file[dataset]["header"]["file"].attrs)
                for key, value in filetype_data.items():
                    config_values[key].append(value)
            except KeyError:
                continue
        if all(len(set(v)) == 1 for v in config_values.values()):
            dtype = config_values["data_type"][0]
            return lambda *args, **kwargs: SimulationCollection(dtype, *args, **kwargs)
        else:
            raise ValueError(
                "Unknown file type. "
                "It appears to have multiple datasets, but organized incorrectly"
            )
    else:
        raise ValueError(
            "Unknown file type. "
            "It appears to have multiple datasets, but organized incorrectly"
        )


class DataCollection(dict):
    """
    A collection of datasets that are related in some way. Provides
    access to high-level operations such as cross-matching, or plotting
    multiple datasets together.


    In general, we want to discourage users from creating their
    own data collections (unless they are derived from one of ours)
    because

    """

    def __init__(
        self,
        collection_type: str,
        header: Optional[OpenCosmoHeader] = None,
        *args,
        **kwargs,
    ):
        self.collection_type = collection_type
        self.__header = header
        super().__init__(*args, **kwargs)

    @property
    def header(self):
        return self.__header

    def __iter__(self):
        # This is more logial than iterating over the keys
        return iter(self.values())

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        for dataset in self.values():
            try:
                dataset.close()
            except ValueError:
                continue

    def write(self, file: h5py.File):
        """
        Write the collection to an HDF5 file.
        """
        # figure out if we have unique headers

        if self._header is None:
            for key, dataset in self.items():
                dataset.write(file, key)
        else:
            self._header.write(file)
            for key, dataset in self.items():
                dataset.write(file, key, with_header=False)


class FileHandle:
    def __init__(self, path: Path):
        self.handle = h5py.File(path, "r")
        self.header = read_header(self.handle)


class LinkedCollection(DataCollection):
    """
    A collection of datasets that are linked together, allowing
    for cross-matching and other operations to be performed.
    """
    def __init__(self, header: OpenCosmoHeader, datasets: dict, links: dict, *args, **kwargs):
        """
        Initialize a linked collection with the provided datasets and links.
        """
        super().__init__("linked", header=header, *args, **kwargs)
        self.links = links
        self.update(datasets)

    @classmethod
    def from_files(cls, *files: Path):
        """
        Given a list of file paths, open them to create a linked collection.
        This is always done out-of-memory.
        """

        file_handles = [FileHandle(file) for file in files]
        datasets = [oc.open(file) for file in files]
        link_spec = verify_links(*[fh.header for fh in file_handles])
        links = {}
        for file_type in link_spec:
            handle = next(filter(lambda x: x.header.file.data_type == file_type, file_handles)).handle
            links[file_type] = get_links(handle)
        if not links:
            raise ValueError("No valid links found in files")

        datasets = {ds.header.file.data_type: ds for ds in datasets}
        return cls(file_handles[0].header, datasets, links)

class SimulationCollection(DataCollection):
    """
    A collection of datasets of the same type from different
    simulations. In general this exposes the exact same API
    as the individual datasets, but maps the results across
    all of them.
    """

    def __init__(self, dtype: str, *args, **kwargs):
        self.dtype = dtype
        super().__init__("multi_simulation", *args, **kwargs)

    def __map(self, method, *args, **kwargs):
        """
        This type of collection will only ever be constructed if all the underlying
        datasets have the same data type, so it is always safe to map operations
        across all of them.
        """
        output = {k: getattr(v, method)(*args, **kwargs) for k, v in self.items()}
        return SimulationCollection(self.dtype, header=self._header, **output)

    def __getattr__(self, name):
        # check if the method exists on the first dataset
        if hasattr(next(iter(self.values())), name):
            return lambda *args, **kwargs: self.__map(name, *args, **kwargs)
        else:
            raise AttributeError(f"Attribute {name} not found on {self.dtype} dataset")


class ParticleCollection(DataCollection):
    """
    A collection of different particle species from the same
    halo.
    """

    def __init__(self, *args, **kwargs):
        super().__init__("particle", *args, **kwargs)

    def collect(self):
        data = {k: v.collect() for k, v in self.items()}
        return ParticleCollection(header=self._header, **data)
