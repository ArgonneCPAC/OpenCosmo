from functools import cached_property

import h5py

from opencosmo import cosmology as cosmo
from opencosmo import parameters
from opencosmo.file import file_reader


class OpenCosmoHeader:
    def __init__(
        self,
        cosmology_pars: parameters.CosmologyParameters,
        simulation_pars: parameters.SimulationParameters,
        reformat_pars: parameters.ReformatParamters,
    ):
        self.__cosmology_pars = cosmology_pars
        self.__simulation_pars = simulation_pars
        self.__reformat_pars = reformat_pars

    @cached_property
    def cosmology(self):
        return cosmo.make_cosmology(self.__cosmology_pars)


@file_reader
def read_header(file: h5py.File) -> OpenCosmoHeader:
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
        reformat_paramters = parameters.read_header_attributes(
            file, "reformat_hacc/config", parameters.ReformatParamters
        )
    except KeyError:
        raise KeyError("File hader is malformedAre you sure it is an OpenCosmo file?")
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
    return OpenCosmoHeader(
        cosmology_paramters, simulation_parameters, reformat_paramters
    )
