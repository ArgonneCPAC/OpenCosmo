from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional
from pathlib import Path
from contextlib import contextmanager

import h5py
import numpy as np


from opencosmo.header import OpenCosmoHeader, read_header
from opencosmo.dataset.mask import Mask
import opencosmo as oc
from .link import verify_links, get_links

class FileHandle:
    """
    Helper class used just for setup
    """
    def __init__(self, path: Path):
        self.handle = h5py.File(path, "r")
        self.header = read_header(self.handle)

def open_linked(*files: Path):
    """
    Open a collection of files that are linked together, such as a
    properties file and a particle file.
    """
    file_handles = [FileHandle(file) for file in files]
    datasets = [oc.open(file) for file in files]
    property_file_type, linked_files = verify_links(*[fh.header for fh in file_handles])
    property_handle = next(filter(lambda x: x.header.file.data_type == property_file_type, file_handles)).handle
    links = get_links(property_handle)
    if not links:
        raise ValueError("No valid links found in files")
    
    output_datasets = {}
    for dataset in datasets:
        if isinstance(dataset, oc.DataCollection):
            output_datasets.update(dataset)
        else:   
            output_datasets[dataset.header.file.data_type] = dataset

    properties_file = output_datasets.pop(property_file_type)
    links = {k: links[k] for k in output_datasets}
    return LinkedCollection(property_handle, properties_file, output_datasets, links)


def get_collection_type(file: h5py.File) -> Callable[..., DataCollection]:
    """
    Determine the type of a single file containing multiple datasets. Currently
    we only support multi_simulation and particle.

    multi_simulation == multiple simulations, same data types
    particle == single simulation, multiple particle species
    """
    datasets = [k for k in file.keys() if k != "header"]
    if len(datasets) == 0:
        raise ValueError("No datasets found in file.")

    if all("particle" in dataset for dataset in datasets) and "header" in file.keys():
        return lambda *args, **kwargs: DataCollection("particle", *args, **kwargs)

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
            return lambda *args, **kwargs: SimulationCollection("particle", *args, **kwargs)
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



class LinkedCollection(DataCollection):
    """
    A collection of datasets that are linked together, allowing
    for cross-matching and other operations to be performed.

    For now, these are always a combination of a properties dataset
    and some other set of datasetes (particles and/or profiles).
    """
    def __init__(self, header: OpenCosmoHeader, properties: oc.Dataset, datasets: dict, links: dict, *args, **kwargs):
        """
        Initialize a linked collection with the provided datasets and links.
        """
        super().__init__("linked", header, *args, **kwargs)
        self.__properties = properties
        self.__datasets = datasets
        self[properties.header.file.data_type] = properties
        self.__linked = links
        self.__idxs = np.where(self.__properties.mask)[0]
        self.update(self.__datasets)

    def __get_linked(self, dtype: str, index: int):
        if dtype not in self.__linked:
            raise ValueError(f"No links found for {dtype}")
        elif index >= len(self.__properties):
            raise ValueError(f"Index {index} out of range for {dtype}")
        # find the index into the linked dataset at the mask index
        linked_index = self.__idxs[index]
        try: 
            start = self.__linked[dtype]["start_index"][linked_index]
            size = self.__linked[dtype]["length"][linked_index]
        except IndexError:
            start = self.__linked[dtype][linked_index]
            size = 1
        
        if start == -1 or size == -1:
            return None
        return self.__datasets[dtype].get_range(start, start + size)

    def items(self, dtypes: Optional[str | list[str]] = None):
        """
        Iterate over the datasets in the collection that are linked to the
        provided data types. This is a generator that yields the linked
        datasets.
        """
    
        
        if dtypes is None:
            dtypes = list(self.__linked.keys())
        elif isinstance(dtypes, str):
            dtypes = [dtypes]
        if not all(dtype in self.__linked for dtype in dtypes):
            raise ValueError("One or more of the provided data types is not linked")
        
        ndtypes = len(dtypes)
        for i, properties in enumerate(self.__properties.rows()):
            results = {dtype: self.__get_linked(dtype, i) for dtype in dtypes}
            if all(result is None for result in results.values()):
                continue
            if ndtypes == 1:
                yield properties, results[dtypes[0]]
            else:
                yield properties, results

    def filter(self, *masks: Mask):
        """
        Filter the datasets in the collection by the provided masks. Filtering occurs
        on the properties file that is linked to the other datasets.
        """
        new_properties = self.__properties.filter(*masks)
        return LinkedCollection(self.header, new_properties, self.__datasets, self.__linked)
    
    def take(self, n: int, at: str = "start"):
        new_properties = self.__properties.take(n, at)
        return LinkedCollection(self.header, new_properties, self.__datasets, self.__linked)
    
    def with_units(self, convention: str) -> LinkedCollection:
        """
        Return a new dataset with the units converted to the provided convention.
        """
        new_properties = self.__properties.with_units(convention)
        new_datasets = {k: v.with_units(convention) for k, v in self.__datasets.items()}
        return LinkedCollection(self.header, new_properties, new_datasets, self.__linked)
    

    
    
    
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


