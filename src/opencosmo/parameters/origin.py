from opencosmo.parameters import hacc

from .cosmology import CosmologyParameters


def get_origin_parameters(origin: str):
    if origin == "HACC":
        return hacc.ORIGIN_PARAMETERS
    else:
        return {"simulation/cosmology": CosmologyParameters}
