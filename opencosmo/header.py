from functools import cached_property
from typing import Optional

import h5py

from opencosmo import cosmology as cosmo
from opencosmo import parameters
from opencosmo.file import broadcast_read, file_reader, file_writer

try:
    from mpi4py import MPI
except ImportError:
    MPI = None  # type: ignore


class OpenCosmoHeader:
    def __init__(
        self,
        file_pars: parameters.FileParameters,
        simulation_pars: parameters.SimulationParameters,
        reformat_pars: parameters.ReformatParamters,
        cosmotools_pars: parameters.CosmoToolsParameters,
    ):
        self.__file_pars = file_pars
        self.__simulation_pars = simulation_pars
        self.__reformat_pars = reformat_pars
        self.__cosmotools_pars = cosmotools_pars

    def write(self, file: h5py.File | h5py.Group) -> None:
        parameters.write_header_attributes(file, "file", self.__file_pars)

        parameters.write_header_attributes(
            file, "reformat_hacc/config", self.__reformat_pars
        )
        parameters.write_header_attributes(
            file, "simulation/parameters", self.__simulation_pars
        )
        parameters.write_header_attributes(
            file, "simulation/cosmotools", self.__cosmotools_pars
        )
        parameters.write_header_attributes(
            file, "simulation/cosmology", self.__simulation_pars.cosmology_parameters
        )
        if hasattr(self.__simulation_pars, "subgrid_parameters"):
            parameters.write_header_attributes(
                file, "simulation/parameters", self.__simulation_pars.subgrid_parameters
            )

    @cached_property
    def cosmology(self):
        return cosmo.make_cosmology(self.__simulation_pars.cosmology_parameters)

    @property
    def simulation(self):
        return self.__simulation_pars

    @property
    def reformat(self):
        return self.__reformat_pars


@file_writer
def write_header(
    file: h5py.File, header: OpenCosmoHeader, dataset_name: Optional[str] = None
) -> None:
    """
    Write the header of an OpenCosmo file

    Parameters
    ----------
    file : h5py.File
        The file to write to
    header : OpenCosmoHeader
        The header information to write

    """
    if dataset_name is not None:
        group = file.require_group(dataset_name)
    else:
        group = file
    header.write(group)


@broadcast_read
@file_reader
def read_header(file: h5py.File | h5py.Group) -> OpenCosmoHeader:
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
        file_parameters = parameters.read_header_attributes(
            file, "file", parameters.FileParameters
        )
    except KeyError as e:
        raise KeyError(
            "File header is malformed. Are you sure it is an OpenCosmo file?\n "
            f"Error: {e}"
        )
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
        file_parameters,
        simulation_parameters,
        reformat_parameters,
        cosmotools_parameters,
    )
