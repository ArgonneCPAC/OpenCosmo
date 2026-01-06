from __future__ import annotations

from copy import copy
from functools import cache
from itertools import chain
from types import UnionType
from typing import TYPE_CHECKING, Any, Optional

import h5py
from pydantic import BaseModel, ValidationError

from opencosmo.file import broadcast_read, file_reader, file_writer
from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.parameters import (
    FileParameters,
    dtype,
    origin,
    read_header_attributes,
    write_header_attributes,
)
from opencosmo.parameters.units import apply_units
from opencosmo.units import UnitConvention

if TYPE_CHECKING:
    from pathlib import Path

    from opencosmo.io.schema import Schema


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
        file_pars: FileParameters,
        required_origin_parameters: dict[str, BaseModel],
        optional_origin_parameters: dict[str, BaseModel],
        dtype_parameters: dict[str, BaseModel],
        unit_convention: UnitConvention = UnitConvention.SCALEFREE,
    ):
        self.__file_pars = file_pars
        self.__required_origin_parameters = required_origin_parameters
        self.__optional_origin_parameters = optional_origin_parameters
        self.__dtype_parameters = dtype_parameters
        self.unit_convention = unit_convention

    def __eq__(self, other):
        return (
            self.__file_pars == other.__file_pars
            and self.__required_origin_parameters == other.__required_origin_parameters
            and self.__optional_origin_parameters == other.__optional_origin_parameters
            and self.__dtype_parameters == other.__dtype_parameters
        )

    def __hash__(self):
        # Create a frozenset of the items in the dictionary
        # Each item is a tuple of (key, hash of the model)
        iter_ = chain(
            {"file": self.__file_pars}.items(),
            {"unit_convention": self.unit_convention}.items(),
            self.__required_origin_parameters.items(),
            self.__optional_origin_parameters.items(),
            self.__dtype_parameters.items(),
        )
        s = frozenset((key, hash(model)) for key, model in iter_)
        return hash(s)

    def with_units(self, convention: UnitConvention | str):
        convention = UnitConvention(convention)
        if convention == self.unit_convention:
            return self
        return OpenCosmoHeader(
            self.__file_pars,
            self.__required_origin_parameters,
            self.__optional_origin_parameters,
            self.__dtype_parameters,
            convention,
        )

    @cache
    def __get_access_table(self):
        table = {}
        all_models = chain(
            self.__required_origin_parameters.values(),
            self.__optional_origin_parameters.values(),
            self.__dtype_parameters.values(),
        )
        for model in all_models:
            if not hasattr(model, "ACCESS_PATH"):
                continue
            table[model.ACCESS_PATH] = model
            if hasattr(model, "ACCESS_TRANSFORMATION"):
                table[model.ACCESS_PATH] = model.ACCESS_TRANSFORMATION()

        cosmology = table.get("cosmology")
        convention = object.__getattribute__(self, "unit_convention")
        scale_factor = None
        if self.__file_pars.redshift is not None:
            scale_factor = cosmology.scale_factor(self.__file_pars.redshift)
        for name, model in table.items():
            if isinstance(model, BaseModel):
                table[name] = apply_units(
                    model,
                    cosmology,
                    convention,
                    unit_kwargs={"scale_factor": scale_factor},
                )

        return table

    @property
    def parameters(self):
        """
        Return the parametrs stored in this header as
        key-value pairs. The values will be Pydantic models.

        Any block of parameters that is returned from this method can
        also be accessed with standard dot notation. For example, HACC
        data contains a "simulation" block that contains the parameters that
        were used to run the original simulation. The following calls are
        equivalent:

        .. code-block:: python

            header.simulation
            header.parameters["simulation"]

        Returns
        -------
        parameters: dict[str, pydantic.BaseModel]
            The parameter blocks associated with this header

        """
        return self.__get_access_table()

    def __getattr__(self, key: str):
        if key.startswith("__"):  # avoid infinite recursion when serailizing for MPI
            raise AttributeError(key)

        table = self.__get_access_table()
        try:
            return table[key]
        except KeyError:
            return object.__getattribute__(self, key)

    def with_region(self, region):
        if region is not None:
            region_model = region.into_model()
        else:
            region_model = None
        new_file_pars = self.__file_pars.model_copy(update={"region": region_model})
        new_header = OpenCosmoHeader(
            new_file_pars,
            self.__required_origin_parameters,
            self.__optional_origin_parameters,
            self.__dtype_parameters,
        )
        return new_header

    def with_parameter(self, key: str, value: Any):
        """
        Update a dtype parameter with a new value. This in general should never
        be called by the user. Returns a copy.
        """
        path = key.split("/")
        if len(path) != 2:
            raise ValueError("Can only update top-level dtype parameters")
        new_dtype_parameters = copy(self.__dtype_parameters)
        model = new_dtype_parameters[path[0]]
        new_model = model.model_copy(update={path[1]: value})
        new_dtype_parameters[path[0]] = new_model
        return OpenCosmoHeader(
            self.__file_pars,
            self.__required_origin_parameters,
            self.__optional_origin_parameters,
            new_dtype_parameters,
        )

    def with_parameters(self, updates: dict[str, Any]):
        if not updates:
            return self
        new_header = self
        for key, val in updates.items():
            new_header = new_header.with_parameter(key, val)
        return new_header

    def dump(self) -> Schema:
        to_write = chain(
            [("file", self.__file_pars)],
            self.__required_origin_parameters.items(),
            self.__optional_origin_parameters.items(),
            self.__dtype_parameters.items(),
        )
        pars = {}
        for path, model in to_write:
            data = model.model_dump(by_alias=True)
            data = dict(
                map(
                    lambda kv: (kv[0], kv[1] if kv[1] is not None else ""), data.items()
                )
            )

            pars[path] = data
        return make_schema("header", FileEntry.METADATA, attributes=pars)

    def write(self, file: h5py.File | h5py.Group) -> None:
        write_header_attributes(file, "file", self.__file_pars)
        to_write = chain(
            self.__required_origin_parameters.items(),
            self.__optional_origin_parameters.items(),
            self.__dtype_parameters.items(),
        )
        for path, data in to_write:
            write_header_attributes(file, path, data)

    @property
    def file(self) -> FileParameters:
        """
        All files must at minimum have a "file" block in their header. This block
        contains basic information like the original source of the data and
        its data type.
        """
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
def read_header(
    file: h5py.File | h5py.Group,
    unit_convention: UnitConvention = UnitConvention.COMOVING,
) -> OpenCosmoHeader:
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
        file_parameters = read_header_attributes(file, "file", FileParameters)
    except KeyError as e:
        raise KeyError(
            "File header is malformed. Are you sure it is an OpenCosmo file?\n "
            f"Error: {e}"
        )

    origin_parameter_models = origin.get_origin_parameters(file_parameters.origin)
    required_origin_params, optional_origin_params = read_origin_parameters(
        file, origin_parameter_models
    )
    dtype_parameter_models = dtype.get_dtype_parameters(file_parameters)
    dtype_params = read_dtype_parameters(file, dtype_parameter_models)

    h = OpenCosmoHeader(
        file_parameters,
        required_origin_params,
        optional_origin_params,
        dtype_params,
        unit_convention,
    )
    return h


def read_origin_parameters(
    file: h5py.File | h5py.Group,
    origin_parameters: dict[str, dict[str, type[BaseModel]]],
):
    """
    An "origin" describes the original source of a given dataset. Currently the only
    origin we support in the OpenCosmo toolkit is HACC.

    Origins can define a set of required and optional parameters.
    """
    required = origin_parameters["required"]
    required_output = {}
    for path, model in required.items():
        if isinstance(model, UnionType):
            required_output[path] = load_union_model(file, path, model)
        else:
            required_output[path] = read_header_attributes(file, path, model)

    optional_output = {}
    optional = origin_parameters["optional"]
    for path, model in optional.items():
        if isinstance(origin, UnionType):
            read_fn = load_union_model
        else:
            read_fn = read_header_attributes
        try:
            optional_output[path] = read_fn(file, path, model)
        except (ValidationError, KeyError):
            continue

    return required_output, optional_output


def read_dtype_parameters(
    file: h5py.File | h5py.Group, dtype_paramter_models: dict[str, type[BaseModel]]
):
    """
    Data types can also define parameters that they expect. For now, all dtype
    parameters are required. They MUST define an "ACCESS_PATH" attribute,
    which tells the header how users should be allowed to access them.

    """
    dtype_output = {}
    for path, model in dtype_paramter_models.items():
        if isinstance(model, UnionType):
            dtype_output[path] = load_union_model(file, path, model)
        else:
            dtype_output[path] = read_header_attributes(file, path, model)

        if not hasattr(model, "ACCESS_PATH"):
            # This should be always always always caught in testing
            raise ValueError(f"Model {model} does not have an access path!")

    return dtype_output


def load_union_model(
    file: h5py.File | h5py.Group, path: str, allowed_models: UnionType, **kwargs
):
    for model in allowed_models.__args__:
        try:
            return read_header_attributes(file, path, model)
        except ValidationError as ve:
            if any(e["type"] == "missing" or e["input"] is None for e in ve.errors()):
                continue
            else:
                raise ValueError(
                    f"Parsing header paramter model raised a validation error: \n {ve}"
                )
    raise ValueError("Input attributes do not match any of the models in the union")
