from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, PositiveInt  # noqa: TC002

from opencosmo.dtypes.parameters import read_header_attributes

if TYPE_CHECKING:
    import h5py


class MapHeaderParameters(BaseModel):
    """
    Minimal, dataset-agnostic header identifying a mapping file as an OpenCosmo
    file.

    A mapping describes the row-level correspondence *between* datasets in
    different simulations and is owned by none of them, so it carries none of
    the cosmology, simulation, origin, or data-type information an
    ``OpenCosmoHeader`` does. This header exists only so that opening a mapping
    file can verify it is an OpenCosmo file, using the same contract every other
    path relies on. It is validated at discovery and then discarded — nothing
    downstream retains it.
    """

    model_config = ConfigDict(frozen=True)
    mapping_version: PositiveInt
    simulation_suite: str


def read_map_header(file: h5py.File | h5py.Group) -> MapHeaderParameters:
    """
    Read and validate the header of a mapping file.

    Reuses the model-agnostic ``read_header_attributes`` primitive to pull the
    attributes at ``/header/map`` and validate them against
    ``MapHeaderParameters``. This deliberately bypasses the origin/data-type
    orchestration in ``read_header``: a mapping file has neither.

    Raises ``KeyError`` if the file has no ``/header`` group and
    ``pydantic.ValidationError`` if the attributes are missing or malformed —
    either of which means the file is not a valid OpenCosmo mapping file.
    """
    return read_header_attributes(file, "map_params", MapHeaderParameters)
