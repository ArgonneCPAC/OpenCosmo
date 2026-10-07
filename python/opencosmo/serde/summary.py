"""Metadata-only serialization for OpenCosmo results."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Literal, Protocol, TypeAlias, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from opencosmo.collection.lightcone.healpix_map import HealpixMap
from opencosmo.collection.lightcone.lightcone import Lightcone
from opencosmo.collection.simulation.simulation import SimulationCollection
from opencosmo.collection.structure.structure import StructureCollection
from opencosmo.dataset.dataset import Dataset
from opencosmo.dtypes.file import DatasetType
from opencosmo.spatial.models import (
    BoxRegionModel,
    ConeRegionModel,
    SkyboxRegionModel,
)
from opencosmo.spatial.region import HealpixRegion
from opencosmo.units import UnitConvention

if TYPE_CHECKING:
    from collections.abc import Iterable

    import astropy.units as u

    from opencosmo.header import OpenCosmoHeader
    from opencosmo.spatial.protocols import Region


class SummaryModel(BaseModel):
    """Base configuration for serialized result metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ColumnSummary(SummaryModel):
    """Metadata describing a visible column."""

    name: str
    description: str | None
    unit: str | None


class HeaderSummary(SummaryModel):
    """Essential metadata from an OpenCosmo header."""

    origin: str
    data_type: DatasetType
    is_lightcone: bool
    redshift: float | None
    step: int | None
    file_unit_convention: UnitConvention


class BoxRegionSummary(SummaryModel):
    """Serialized three-dimensional box coverage."""

    kind: Literal["box"] = "box"
    p1: tuple[float, float, float]
    p2: tuple[float, float, float]


class ConeRegionSummary(SummaryModel):
    """Serialized cone coverage in degrees."""

    kind: Literal["cone"] = "cone"
    center: tuple[float, float]
    radius: float


class SkyboxRegionSummary(SummaryModel):
    """Serialized sky-box coverage in degrees."""

    kind: Literal["skybox"] = "skybox"
    p1: tuple[float, float]
    p2: tuple[float, float]


class HealpixRegionSummary(SummaryModel):
    """Compact HEALPix coverage without individual pixel identifiers."""

    kind: Literal["healpix"] = "healpix"
    nside: int = Field(gt=0)
    pixel_count: int = Field(ge=0)


RegionSummary: TypeAlias = Annotated[
    BoxRegionSummary | ConeRegionSummary | SkyboxRegionSummary | HealpixRegionSummary,
    Field(discriminator="kind"),
]


class HealpixMetadata(SummaryModel):
    """Essential HEALPix map metadata."""

    nside: int = Field(gt=0)
    nside_lr: int = Field(gt=0)
    ordering: str
    full_sky: bool
    pixel_count: int = Field(ge=0)


class DatasetSummary(SummaryModel):
    """Metadata describing a dataset without its row data."""

    kind: Literal["dataset"] = "dataset"
    uuid: UUID | None = None
    length: int = Field(ge=0)
    columns: tuple[ColumnSummary, ...]
    header: HeaderSummary
    current_unit_convention: UnitConvention
    sorted_by: str | None
    region: RegionSummary | None


class LightconeSummary(SummaryModel):
    """Metadata describing a lightcone and its members."""

    kind: Literal["lightcone"] = "lightcone"
    uuid: UUID | None = None
    length: int = Field(ge=0)
    columns: tuple[ColumnSummary, ...]
    header: HeaderSummary
    current_unit_convention: UnitConvention
    sorted_by: str | None
    region: RegionSummary | None
    redshift_range: tuple[float, float]
    members: tuple[CollectionMember, ...]
    map: HealpixMapSummary | None = None


class HealpixMapSummary(SummaryModel):
    """Metadata describing a HEALPix map and its layers."""

    kind: Literal["healpix_map"] = "healpix_map"
    uuid: UUID | None = None
    length: int = Field(ge=0)
    columns: tuple[ColumnSummary, ...]
    header: HeaderSummary
    current_unit_convention: UnitConvention
    region: RegionSummary | None
    redshift_range: tuple[float, float]
    healpix: HealpixMetadata
    members: tuple[CollectionMember, ...]


class StructureCollectionSummary(SummaryModel):
    """Metadata describing a structure collection and its datasets."""

    kind: Literal["structure_collection"] = "structure_collection"
    uuid: UUID | None = None
    length: int = Field(ge=0)
    data_type: DatasetType
    header: HeaderSummary
    region: RegionSummary | None
    sorted_by: str | None
    members: tuple[CollectionMember, ...]


class SimulationCollectionSummary(SummaryModel):
    """Metadata describing a collection of simulation scopes."""

    kind: Literal["simulation_collection"] = "simulation_collection"
    uuid: UUID | None = None
    length: int = Field(ge=0)
    members: tuple[CollectionMember, ...]


type ResultSummary = Annotated[
    DatasetSummary
    | LightconeSummary
    | HealpixMapSummary
    | StructureCollectionSummary
    | SimulationCollectionSummary,
    Field(discriminator="kind"),
]


class CollectionMember(SummaryModel):
    """A keyed member of a serialized collection."""

    key: str
    value: ResultSummary


type SerializableResult = (
    Dataset | Lightcone | HealpixMap | StructureCollection | SimulationCollection
)
type CollectionKey = str | int | float
type UUIDResolver = Callable[[SerializableResult], UUID | None]


class _HeaderResult(Protocol):
    @property
    def header(self) -> OpenCosmoHeader: ...


class _CollectionResult(Protocol):
    def items(self) -> Iterable[tuple[CollectionKey, SerializableResult]]: ...


def _dataset_type(value: DatasetType | str) -> DatasetType:
    if isinstance(value, DatasetType):
        return value
    return DatasetType(cast("str", value))


def _unit_convention(value: UnitConvention | str) -> UnitConvention:
    if isinstance(value, UnitConvention):
        return value
    return UnitConvention(cast("str", value))


def _header(value: SerializableResult) -> HeaderSummary:
    file = cast("_HeaderResult", value).header.file
    return HeaderSummary(
        origin=file.origin,
        data_type=_dataset_type(file.data_type),
        is_lightcone=file.is_lightcone,
        redshift=file.redshift,
        step=file.step,
        file_unit_convention=_unit_convention(file.unit_convention),
    )


def _columns(
    names: list[str],
    descriptions: dict[str, str | None],
    units: dict[str, u.Unit | None],
) -> tuple[ColumnSummary, ...]:
    return tuple(
        ColumnSummary(
            name=name,
            description=descriptions.get(name),
            unit=None if units.get(name) is None else str(units[name]),
        )
        for name in names
    )


def _region(value: Region | None) -> RegionSummary | None:
    if value is None:
        return None
    if isinstance(value, HealpixRegion):
        return HealpixRegionSummary(
            nside=value.nside,
            pixel_count=len(value.pixels),
        )

    model = value.into_model()
    if model is None:
        return None
    if isinstance(model, BoxRegionModel):
        return BoxRegionSummary(p1=model.p1, p2=model.p2)
    if isinstance(model, ConeRegionModel):
        return ConeRegionSummary(center=model.center, radius=model.radius)
    if isinstance(model, SkyboxRegionModel):
        return SkyboxRegionSummary(p1=model.p1, p2=model.p2)
    raise TypeError(f"Unsupported region type: {type(value).__name__}")


def _current_convention(value: Dataset | Lightcone | HealpixMap) -> UnitConvention:
    if isinstance(value, Dataset):
        return _unit_convention(value._state.convention)
    child = next(iter(value.values()))
    return _current_convention(child)


def _uuid(value: SerializableResult, resolver: UUIDResolver | None) -> UUID | None:
    return None if resolver is None else resolver(value)


def _members(
    values: _CollectionResult, resolver: UUIDResolver | None
) -> tuple[CollectionMember, ...]:
    items = values.items()
    return tuple(
        CollectionMember(
            key=str(key),
            value=serialize_result(child, resolve_uuid=resolver),
        )
        for key, child in items
    )


def serialize_result(
    value: SerializableResult,
    *,
    resolve_uuid: UUIDResolver | None = None,
) -> ResultSummary:
    """Serialize metadata about a dataset or collection.

    This function does not materialize column values. Collection traversal may invoke
    the collection's normal lazy child-rebuilding behavior.

    Parameters
    ----------
    value
        Dataset or collection to describe.
    resolve_uuid
        Optional callback that assigns an opaque runtime UUID to each serialized node.

    Returns
    -------
    ResultSummary
        A validated, metadata-only description of ``value``.

    Raises
    ------
    TypeError
        If ``value`` is not a supported OpenCosmo result.
    """
    if isinstance(value, Dataset):
        return DatasetSummary(
            uuid=_uuid(value, resolve_uuid),
            length=len(value),
            columns=_columns(value.columns, value.descriptions, value.units),
            header=_header(value),
            current_unit_convention=_current_convention(value),
            sorted_by=value.sorted_by,
            region=_region(value.region),
        )
    if isinstance(value, Lightcone):
        return LightconeSummary(
            uuid=_uuid(value, resolve_uuid),
            length=len(value),
            columns=_columns(value.columns, value.descriptions, value.units),
            header=_header(value),
            current_unit_convention=_current_convention(value),
            sorted_by=value.sorted_by,
            region=_region(value.region),
            redshift_range=value.z_range,
            members=_members(value, resolve_uuid),
            map=_serialize_map(value.map, resolve_uuid),
        )
    if isinstance(value, HealpixMap):
        child = next(iter(value.values()))
        return HealpixMapSummary(
            uuid=_uuid(value, resolve_uuid),
            length=len(value),
            columns=_columns(value.columns, value.descriptions, child.units),
            header=_header(value),
            current_unit_convention=_current_convention(value),
            region=_region(value.region),
            redshift_range=value.z_range,
            healpix=HealpixMetadata(
                nside=value.nside,
                nside_lr=value.nside_lr,
                ordering=value.ordering,
                full_sky=value.full_sky,
                pixel_count=len(value.pixels),
            ),
            members=_members(value, resolve_uuid),
        )
    if isinstance(value, StructureCollection):
        return StructureCollectionSummary(
            uuid=_uuid(value, resolve_uuid),
            length=len(value),
            data_type=_dataset_type(value.dtype),
            header=_header(value),
            region=_region(value.region),
            sorted_by=value.sorted_by,
            members=_members(value, resolve_uuid),
        )
    if isinstance(value, SimulationCollection):
        return SimulationCollectionSummary(
            uuid=_uuid(value, resolve_uuid),
            length=len(value),
            members=_members(value, resolve_uuid),
        )
    raise TypeError(f"Unsupported result type: {type(value).__name__}")


def _serialize_map(
    value: HealpixMap | None, resolver: UUIDResolver | None
) -> HealpixMapSummary | None:
    if value is None:
        return None
    summary = serialize_result(value, resolve_uuid=resolver)
    if not isinstance(summary, HealpixMapSummary):
        raise RuntimeError("HEALPix map serialization produced an invalid result")
    return summary


CollectionMember.model_rebuild()


__all__ = [
    "BoxRegionSummary",
    "CollectionKey",
    "CollectionMember",
    "ColumnSummary",
    "ConeRegionSummary",
    "DatasetSummary",
    "HeaderSummary",
    "HealpixMapSummary",
    "HealpixMetadata",
    "HealpixRegionSummary",
    "LightconeSummary",
    "RegionSummary",
    "ResultSummary",
    "SerializableResult",
    "SimulationCollectionSummary",
    "SkyboxRegionSummary",
    "StructureCollectionSummary",
    "UUIDResolver",
    "serialize_result",
]
