import h5py

from opencosmo import parameters
from opencosmo.file import file_reader


@file_reader
def read_header(file: h5py.File) -> parameters.SimulationParameters:
    try:
        cosmology_paramters = parameters.read_header_attributes(
            file, "simulation/cosmology", parameters.CosmologyParameters
        )
    except KeyError:
        raise KeyError(
            "This file does not appear to have cosmology information. "
            "Are you sure it is an OpenCosmo file?"
        )
    try:
        simulation_parameters = parameters.read_header_attributes(
            file,
            "simulation/parameters",
            parameters.SimulationParameters,
            cosmology=cosmology_paramters,
        )
    except KeyError:
        raise KeyError(
            "This file does not appear to have simulation information. "
            "Are you sure it is an OpenCosmo file?"
        )
    return simulation_parameters
