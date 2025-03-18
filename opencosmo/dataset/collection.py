import h5py
from typing import Optional
from opencosmo.header import OpenCosmoHeader
from collections import defaultdict


def get_collection_type(file: h5py.File) -> type:
    """
    Determine the type of a file containing multiple datasets. Currently
    we only support multi_simulation and particle.

    multi_simulation == multiple simulations, same data types
    particle == single simulation, multiple particle species
    """
    datasets = [k for k in file.keys() if k != "header"]
    if len(datasets) == 0:
        raise ValueError('No datasets found in file.')
    
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
            raise ValueError("Unknown file type. It appears to have multiple datasets, but organized incorrectly")
    else:
        raise ValueError("Unknown file type. It appears to have multiple datasets, but organized incorrectly")


class DataCollection(dict):
    """
    A collection of datasets that are related in some way. Provides
    access to high-level operations such as cross-matching, or plotting
    multiple datasets together.


    In general, we want to discourage users from creating their
    own data collections (unless they are derived from one of ours)
    because 

    """
    def __init__(self, collection_type: str, header: Optional[OpenCosmoHeader] = None, *args, **kwargs):
        self.collection_type = collection_type
        self.__header = header
        super().__init__(*args, **kwargs)

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
    
        if self.__header is None:
            for key, dataset in self.items():
                dataset.write(file, key)
        else:
            self.__header.write(file)
            for key, dataset in self.items():
                dataset.write(file, key, with_header=False)

    def collect(self):
        data = {k: v.collect() for k, v in self.items()}
        return DataCollection(header=self.__header, **data)


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


class ParticleCollection(DataCollection):
    """
    A collection of different particle species from the same
    halo.
    """
    def __init__(self, *args, **kwargs):
        super().__init__("particle", *args, **kwargs)




