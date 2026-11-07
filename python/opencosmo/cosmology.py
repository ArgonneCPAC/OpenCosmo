from __future__ import annotations

from typing import TYPE_CHECKING

from opencosmo import header
from opencosmo.file import broadcast_read, file_reader

if TYPE_CHECKING:
    import h5py
    from astropy import cosmology

"""
Reads cosmology from the header of the file and returns the
astropy.cosmology object.
"""


@broadcast_read
@file_reader
def read_cosmology(file: h5py.File | h5py.Group) -> cosmology.Cosmology:
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
