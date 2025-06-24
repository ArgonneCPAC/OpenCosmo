from functools import cached_property
from pathlib import Path
from typing import Optional

import h5py

from opencosmo import cosmology as cosmo
from opencosmo import parameters
from opencosmo.file import broadcast_read, file_reader, file_writer


class OpenCosmoHeader:
    """
    A class to represent the header of an OpenCosmo file. The header contains
    information about the simulation the data is a part of, as well as other
    meatadata that are useful to the library in various contexts. Most files
    will have a single unique header, but it is possible to have multiple
    headers in a SimulationCollection.
    """

    def __init__(
        self,
        file_pars: parameters.FileParameters,
        simulation_pars: parameters.SimulationParameters,
        cosmotools_pars: Optional[parameters.CosmoToolsParameters],
    ):
        self.__file_pars = file_pars
        self.__simulation_pars = simulation_pars
        self.__cosmotools_pars = cosmotools_pars

    def write(self, file: h5py.File | h5py.Group) -> None:
        parameters.write_header_attributes(file, "file", self.__file_pars)

        parameters.write_header_attributes(
            file, "simulation/parameters", self.__simulation_pars
        )
        if self.__cosmotools_pars is not None:
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
    def file(self):
        return self.__file_pars

    @property
    def cosmotools(self):
        return self.__cosmotools_pars


@file_writer
def write_header(
    path: Path, header: OpenCosmoHeader, dataset_name: Optional[str] = None
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
    with h5py.File(path, "w") as f:
        if dataset_name is not None:
            group = f.require_group(dataset_name)
        else:
            group = f
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
        simulation_parameters = parameters.read_simulation_parameters(file)

    except (TypeError, KeyError) as e:
        raise ValueError(
            "This file does not appear to have simulation information, or the "
            "simulation information is malformed. "
            "Are you sure it is an OpenCosmo file?\n"
            f"Error: {e}"
        )

    try:
        cosmotools_parameters = parameters.read_header_attributes(
            file, "simulation/cosmotools", parameters.CosmoToolsParameters
        )
    except KeyError:
        cosmotools_parameters = None
    return OpenCosmoHeader(
        file_parameters,
        simulation_parameters,
        cosmotools_parameters,
    )
