import inspect
from typing import Type

import h5py
from astropy import cosmology

from opencosmo.file import file_reader
from opencosmo.header import read_header
from opencosmo.parameters import CosmologyParameters

"""
Reads cosmology from the header of the file and returns the
astropy.cosmology object.
"""


@file_reader
def read_cosmology(file: h5py.File) -> cosmology.Cosmology:
    """
    Read cosmology from the header of an OpenCosmo file

    This function reads the cosmology parameters from the
    header of an OpenCosmo file and returns the most specific
    astropy.Cosmology object that it can. For example, it can
    distinguish between FlatLambdaCDM and non-flat wCDM models.

    Parameters
    ----------
    file : h5py.File | str | Path
        The open cosmology file or the path to the file

    Returns
    -------
    cosmology : astropy.Cosmology
        The cosmology object corresponding to the cosmology in the file
    """

    parameters = read_header(file)
    return make_cosmology(parameters.cosmology)


def make_cosmology(parameters: CosmologyParameters) -> cosmology.Cosmology:
    cosmology_type = get_cosmology_type(parameters)
    expected_arguments = inspect.signature(cosmology_type).parameters.keys()
    input_paremeters = {}
    for argname in expected_arguments:
        try:
            input_paremeters[argname] = getattr(parameters, argname)
        except AttributeError:
            continue
    return cosmology_type(**input_paremeters)


def get_cosmology_type(parameters: CosmologyParameters) -> Type[cosmology.Cosmology]:
    is_flat = (parameters.Om0 + parameters.Ode0) == 1.0
    if parameters.w0 == -1 and parameters.wa == 0:
        if is_flat:
            return cosmology.FlatLambdaCDM
        else:
            return cosmology.LambdaCDM
    if parameters.w0 != -1 and parameters.wa == 0:
        if is_flat:
            return cosmology.FlatwCDM
        else:
            return cosmology.wCDM
    if parameters.wa != 0:
        if is_flat:
            return cosmology.Flatw0waCDM
        else:
            return cosmology.w0waCDM

    raise ValueError("Could not determine cosmology type.")
