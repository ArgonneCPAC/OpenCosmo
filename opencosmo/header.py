from functools import cached_property
from itertools import chain
from pathlib import Path
from types import UnionType
from typing import Optional

import h5py
from pydantic import BaseModel, ValidationError

from opencosmo import cosmology as cosmo
from opencosmo import parameters
from opencosmo.file import broadcast_read, file_reader, file_writer
from opencosmo.parameters import origin


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
        required_origin_parameters: dict[str, BaseModel],
        optional_origin_parameters: dict[str, BaseModel],
    ):
        self.__file_pars = file_pars
        self.__required_origin_parameters = required_origin_parameters
        self.__optional_origin_parameters = optional_origin_parameters

    def with_region(self, region):
        region_model = region.into_model()
        new_file_pars = self.__file_pars.model_copy(update={"region": region_model})
        new_header = OpenCosmoHeader(
            new_file_pars,
            self.__required_origin_parameters,
            self.__optional_origin_parameters,
        )
        return new_header

    def write(self, file: h5py.File | h5py.Group) -> None:
        parameters.write_header_attributes(file, "file", self.__file_pars)
        to_write = chain(
            self.__required_origin_parameters.items(),
            self.__optional_origin_parameters.items(),
        )
        for path, data in to_write:
            parameters.write_header_attributes(file, path, data)

    @cached_property
    def cosmology(self):
        cosmo_pars = [
            val
            for key, val in self.__required_origin_parameters.items()
            if "cosmology" in key
        ]
        if len(cosmo_pars) != 1:
            raise ValueError(
                "This dataset does not appear to have cosmology information"
            )
        return cosmo.make_cosmology(cosmo_pars[0])

    @property
    def simulation(self):
        all_models = chain(
            self.__required_origin_parameters.values(),
            self.__optional_origin_parameters.values(),
        )
        for model in all_models:
            try:
                if model.ACCESS_PATH == "simulation":
                    return model
            except AttributeError:
                continue
        raise ValueError("This dataset does not appear to have simulation parameters")

    @property
    def file(self):
        return self.__file_pars


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

    origin_parameter_models = origin.get_origin_parameters(file_parameters.origin)
    required_origin_params, optional_origin_params = read_origin_parameters(
        file, origin_parameter_models
    )

    return OpenCosmoHeader(
        file_parameters, required_origin_params, optional_origin_params
    )


def read_origin_parameters(
    file: h5py.File | h5py.Group, origin_parameters: dict[str, dict[str, type]]
):
    required = origin_parameters["required"]
    required_output = {}
    for path, model in required.items():
        if isinstance(model, UnionType):
            required_output[path] = load_union_model(file, path, model)
        else:
            required_output[path] = parameters.read_header_attributes(file, path, model)

    optional_output = {}
    optional = origin_parameters["optional"]
    for path, model in optional.items():
        if isinstance(origin, UnionType):
            read_fn = load_union_model
        else:
            read_fn = parameters.read_header_attributes
        try:
            optional_output[path] = read_fn(file, path, model)
        except (ValidationError, KeyError):
            continue

    return required_output, optional_output


def load_union_model(
    file: h5py.File | h5py.Group, path: str, allowed_models: UnionType, **kwargs
):
    for model in allowed_models.__args__:
        try:
            return parameters.read_header_attributes(file, path, model)
        except ValidationError:
            continue
    raise ValueError("Input attributes do not match any of the models in the union")
