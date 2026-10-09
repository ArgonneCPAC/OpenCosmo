"""Human-readable string representations of result summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .summary import (
    BoxRegionSummary,
    ConeRegionSummary,
    DatasetSummary,
    HealpixMapSummary,
    HealpixRegionSummary,
    LightconeSummary,
    SimulationCollectionSummary,
    SkyboxRegionSummary,
    StructureCollectionSummary,
)

if TYPE_CHECKING:
    from .summary import (
        CollectionMember,
        ColumnSummary,
        RegionSummary,
        ResultSummary,
    )

MAX_COLUMNS = 10


def __format_column(column: ColumnSummary) -> str:
    unit = f" [{column.unit}]" if column.unit else ""
    description = f": {column.description}" if column.description else ""
    return f"  {column.name}{unit}{description}"


def __format_columns(columns: tuple[ColumnSummary, ...]) -> str:
    lines = [f"Columns ({len(columns)}):"]
    lines.extend(__format_column(column) for column in columns[:MAX_COLUMNS])
    if len(columns) > MAX_COLUMNS:
        lines.append(f"  ... and {len(columns) - MAX_COLUMNS} more")
    return "\n".join(lines)


def __format_members(members: tuple[CollectionMember, ...]) -> str:
    lines = [f"Members ({len(members)}):"]
    for member in members:
        first_line = format_summary(member.value).split("\n", maxsplit=1)[0]
        lines.append(f"  {member.key}: {first_line}")
    return "\n".join(lines)


def format_region(region: RegionSummary) -> str:
    """Describe the spatial coverage of a result."""
    match region:
        case BoxRegionSummary():
            return f"Box with bounds {list(zip(region.p1, region.p2))}"
        case ConeRegionSummary():
            ra, dec = region.center
            return (
                f"Cone Region (center: RA={ra:.4f}°, Dec={dec:.4f}°, "
                f"radius={region.radius:.4f}°)"
            )
        case SkyboxRegionSummary():
            ra0, dec0 = region.p1
            ra1, dec1 = region.p2
            return (
                f"Sky Box Region (RA: {ra0:.4f}°–{ra1:.4f}°, "
                f"Dec: {dec0:.4f}°–{dec1:.4f}°)"
            )
        case HealpixRegionSummary():
            return (
                f"Healpix Region (nside = {region.nside}, {region.pixel_count} pixels)"
            )


def __format_details(
    summary: DatasetSummary | LightconeSummary | HealpixMapSummary,
) -> list[str]:
    lines = [__format_columns(summary.columns)]
    lines.append(f"Unit convention: {summary.current_unit_convention.value}")
    if not isinstance(summary, HealpixMapSummary) and summary.sorted_by is not None:
        lines.append(f"Sorted by: {summary.sorted_by}")
    if summary.region is not None:
        lines.append(f"Region: {format_region(summary.region)}")
    return lines


def format_summary(summary: ResultSummary) -> str:
    """Build a user-facing description of a result from its summary.

    The first line always identifies the result type and its length.
    """
    match summary:
        case DatasetSummary():
            head = f"OpenCosmo Dataset (length={summary.length})"
            return "\n".join([head, *__format_details(summary)])
        case LightconeSummary():
            z_low, z_high = summary.redshift_range
            head = (
                f"OpenCosmo Lightcone Dataset (length={summary.length}, "
                f"{z_low} < z < {z_high})"
            )
            lines = [head, *__format_details(summary)]
            if summary.map is not None:
                lines.append(f"Map: nside={summary.map.healpix.nside}")
            lines.append(__format_members(summary.members))
            return "\n".join(lines)
        case HealpixMapSummary():
            z_low, z_high = summary.redshift_range
            healpix = summary.healpix
            head = (
                f"OpenCosmo Healpix Map Dataset (length={summary.length}, "
                f"{z_low} < z < {z_high})"
            )
            coverage = "full sky" if healpix.full_sky else "partial sky"
            lines = [
                head,
                *__format_details(summary),
                f"Healpix: nside={healpix.nside}, ordering={healpix.ordering}, "
                f"{coverage} ({healpix.pixel_count} pixels)",
                __format_members(summary.members),
            ]
            return "\n".join(lines)
        case StructureCollectionSummary():
            structure_type = summary.data_type.value.split("_")[0] + "s"
            lightcone = " on a lightcone" if summary.header.is_lightcone else ""
            head = (
                f"Collection of {structure_type}{lightcone} (length={summary.length})"
            )
            lines = [head]
            if summary.sorted_by is not None:
                lines.append(f"Sorted by: {summary.sorted_by}")
            if summary.region is not None:
                lines.append(f"Region: {format_region(summary.region)}")
            lines.append(__format_members(summary.members))
            return "\n".join(lines)
        case SimulationCollectionSummary():
            n_datasets = sum(
                isinstance(
                    m.value, DatasetSummary | LightconeSummary | HealpixMapSummary
                )
                for m in summary.members
            )
            n_collections = len(summary.members) - n_datasets
            head = (
                f"SimulationCollection({n_collections} collections, "
                f"{n_datasets} datasets)"
            )
            return "\n".join([head, __format_members(summary.members)])


__all__ = ["format_region", "format_summary"]
