import h5py

class DataCollection(dict):
    """
    A collection of datasets that are related in some way. Provides
    access to high-level operations such as cross-matching, or plotting
    multiple datasets together.


    In general, we want to discourage users from creating their
    own data collections (unless they are derived from one of ours)
    because 

    """
    def __init__(self, collection_type: str, *args, **kwargs):
        self.__collection_type = collection_type
        super().__init__(*args, **kwargs)

    def write(self, file: h5py.File):
        """
        Write the collection to an HDF5 file.
        """
        # figure out if we have unique headers
        headers = {ds.__header for ds in self.values()}


        


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




