from functools import cached_property

import h5py

from opencosmo import cosmology as cosmo
from opencosmo import parameters
from opencosmo.file import file_reader


class OpenCosmoHeader:
    def __init__(
        self,
        simulation_pars: parameters.SimulationParameters,
        reformat_pars: parameters.ReformatParamters,
        cosmotools_pars: parameters.CosmoToolsParameters,
    ):
        self.__simulation_pars = simulation_pars
        self.__reformat_pars = reformat_pars
        self.__cosmotools_pars = cosmotools_pars

    @cached_property
    def cosmology(self):
        return cosmo.make_cosmology(self.__simulation_pars.cosmology_parameters)


@file_reader
def read_header(file: h5py.File) -> OpenCosmoHeader:
    """
    Read the header of an OpenCosmo file

    This function may be useful if you just want to access some basic
    information about the simulation but you don't plan to actually
    read any data.

    Parameters
    ----------
    file : str | Path
        The path to the file

    Returns
    -------
    header : OpenCosmoHeader
        The header information from the file


    """
    try:
        reformat_parameters = parameters.read_header_attributes(
            file, "reformat_hacc/config", parameters.ReformatParamters
        )
    except KeyError as e:
        raise KeyError(
            "File header is malformed. Are you sure it is an OpenCosmo file?\n "
            f"Error: {e}"
        )
    try:
        simulation_parameters = parameters.read_simulation_parameters(file)

    except KeyError as e:
        raise KeyError(
            "This file does not appear to have simulation information. "
            "Are you sure it is an OpenCosmo file?\n"
            f"Error: {e}"
        )

    try:
        cosmotools_parameters = parameters.read_header_attributes(
            file, "simulation/cosmotools", parameters.CosmoToolsParameters
        )
    except KeyError as e:
        raise KeyError(
            "This file does not appear to have cosmotools information. "
            "Are you sure it is an OpenCosmo file?\n"
            f"Error: {e}"
        )

    return OpenCosmoHeader(
        simulation_parameters, reformat_parameters, cosmotools_parameters
    )
