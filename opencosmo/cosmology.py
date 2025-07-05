import inspect
from typing import TYPE_CHECKING, Type

import h5py
from astropy import cosmology  # type: ignore

from opencosmo import header
from opencosmo.file import broadcast_read, file_reader

if TYPE_CHECKING:
    from opencosmo.parameters import CosmologyParameters

"""
Reads cosmology from the header of the file and returns the
astropy.cosmology object.
"""


@broadcast_read
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
    file : str | Path
        The path to the file

    Returns
    -------
    cosmology : astropy.Cosmology
        The cosmology object corresponding to the cosmology in the file
    """

    head = header.read_header(file)
    # The header reads parameters and calls into the code
    # below to produce an actual cosmology object.
    return head.cosmology


def make_cosmology(parameters: "CosmologyParameters") -> cosmology.Cosmology:
    cosmology_type = get_cosmology_type(parameters)
    expected_arguments = inspect.signature(cosmology_type).parameters.keys()
    input_paremeters = {}
    for argname in expected_arguments:
        try:
            input_paremeters[argname] = getattr(parameters, argname)
        except AttributeError:
            continue
    return cosmology_type(**input_paremeters)


def get_cosmology_type(parameters: "CosmologyParameters") -> Type[cosmology.Cosmology]:
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
