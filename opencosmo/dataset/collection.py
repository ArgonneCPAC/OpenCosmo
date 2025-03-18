import h5py
from typing import Optional
from opencosmo.header import OpenCosmoHeader

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
    def __init__(self, *args, **kwargs):
        super().__init__("simulation", *args, **kwargs)


class ParticleCollection(DataCollection):
    """
    A collection of different particle species from the same
    halo.
    """
    def __init__(self, *args, **kwargs):
        super().__init__("particle", *args, **kwargs)




