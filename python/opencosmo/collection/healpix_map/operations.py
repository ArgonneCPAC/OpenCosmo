from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import astropy.units as u
import healpy as hp
import healsparse as hsp
import numpy as np

from opencosmo.collection.healpix_map.cutout import (
    get_included_pixels,
    validate_cutout_size,
)
from opencosmo.collection.healpix_map.reproject import make_hdulist
from opencosmo.collection.healpix_map.state import HealpixMapState
from opencosmo.column.column import Column
from opencosmo.dataset import operations as dsops
from opencosmo.dataset import state as st
from opencosmo.io.schema import FileEntry
from opencosmo.io.schema import make_schema as make_map_schema
from opencosmo.mpi import get_comm_world
from opencosmo.spatial import make_skybox
from opencosmo.spatial.index import query_healpix_partitions
from opencosmo.spatial.region import ConeRegion, FullSkyRegion, SkyboxRegion

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    from astropy.coordinates import SkyCoord
    from astropy.io import fits

    from opencosmo.column.column import ColumnMask, ConstructedColumn
    from opencosmo.dataset.state import DatasetState
    from opencosmo.io.schema import Schema
    from opencosmo.spatial.protocols import Region

MapFormat = Literal["healsparse", "healpix", "raw"]


def filter(state: HealpixMapState, *masks: ColumnMask) -> HealpixMapState:
    if not masks:
        return state
    return __with_dataset(state, dsops.filter(state.dataset, *masks, mode="global"))


def take(state: HealpixMapState, n: int, at: str) -> HealpixMapState:
    if n > len(state):
        raise ValueError(
            "Number of rows to take must be less than number of rows in dataset"
        )
    if at not in ("start", "end", "random"):
        raise ValueError(f'"at" should be one of ("start", "end", "random", got {at}')
    return __with_dataset(state, dsops.take(state.dataset, n, at, "local"))


def take_range(state: HealpixMapState, start: int, end: int) -> HealpixMapState:
    if start < 0 or end > len(state):
        raise ValueError("Got row indices that are out of range!")
    return __with_dataset(state, dsops.take_range(state.dataset, start, end, "local"))


def take_rows(state: HealpixMapState, rows: np.ndarray) -> HealpixMapState:
    rows = np.sort(rows)
    if rows[-1] >= len(state) or rows[0] < 0:
        raise ValueError("Rows must be between 0 and the length of this dataset - 1")
    return __with_dataset(state, dsops.take_rows(state.dataset, rows))


def take_rows_unsorted(state: HealpixMapState, rows: np.ndarray) -> HealpixMapState:
    """
    Take rows by their position in raw (pixel) order, ignoring any sort.
    ``rows`` is assumed to be sorted.
    """
    dataset = state.dataset
    taken = st.take_rows(replace(dataset, sort_key=None), rows)
    return __with_dataset(state, replace(taken, sort_key=dataset.sort_key))


def sort_by(state: HealpixMapState, column: str, invert: bool) -> HealpixMapState:
    if column not in state.columns:
        raise ValueError(f"Column {column} does not exist in this dataset!")
    return __with_dataset(state, dsops.sort_by(state.dataset, column, invert))


def bound(
    state: HealpixMapState, region: Region, inclusive: bool = False
) -> HealpixMapState:
    if isinstance(region, SkyboxRegion):
        level = int(np.log2(state.nside))
        pixels = np.union1d(*query_healpix_partitions(region, level)[level])
    elif isinstance(region, ConeRegion):
        vec = hp.ang2vec(region.center.ra.value, region.center.dec.value, lonlat=True)
        pixels = hp.query_disc(
            state.nside,
            vec,
            region.radius.to(u.radian).value,
            inclusive=inclusive,
            nest=state.ordering == "NESTED",
        )
    else:
        raise TypeError("Didn't recieve a 2d region!")

    rows = np.nonzero(np.isin(state.pixels, pixels))[0]
    return take_rows_unsorted(state, rows)


def select(
    state: HealpixMapState,
    *columns: str | Iterable[str],
    **derived_columns: ConstructedColumn,
) -> HealpixMapState:
    return __with_dataset(
        state, dsops.select(state.dataset, *columns, mode="global", **derived_columns)
    )


def drop(state: HealpixMapState, columns: str | Iterable[str]) -> HealpixMapState:
    dropped = {columns} if isinstance(columns, str) else set(columns)
    if missing := dropped.difference(state.columns):
        raise ValueError(
            f"Tried to drop columns that are not in this dataset: {missing}"
        )
    return __with_dataset(state, dsops.drop(state.dataset, dropped))


def with_new_columns(
    state: HealpixMapState,
    descriptions: str | dict[str, str] = {},
    **columns: Column | np.ndarray | u.Quantity,
) -> HealpixMapState:
    for name, column in columns.items():
        if not isinstance(column, Column) and len(column) != len(state):
            raise ValueError(
                f"New column {name} has length {len(column)} but this dataset "
                f"has length {len(state)}"
            )
    return __with_dataset(
        state,
        dsops.with_new_columns(
            state.dataset, descriptions, False, mode="global", **columns
        ),
    )


def evaluate(
    state: HealpixMapState,
    func: Callable,
    format: str = "numpy",
    vectorize: bool = False,
    insert: bool = True,
    **evaluate_kwargs,
) -> HealpixMapState | dict:
    result = dsops.evaluate(
        state.dataset,
        func,
        vectorize=vectorize,
        insert=insert,
        format=format,
        **evaluate_kwargs,
    )
    if not insert:
        return result
    return __with_dataset(state, result)


def rows(state: HealpixMapState) -> Generator[dict[str, float | u.Quantity]]:
    yield from dsops.rows(state.dataset)


def get_data(state: HealpixMapState, format: MapFormat, nside_out: int | None = None):
    if format not in {"healsparse", "healpix", "raw"}:
        raise ValueError(f"Unknown format type {format}")
    if nside_out is not None:
        state = with_resolution(state, nside_out)

    # Pixels follow raw-row order, so the data must ignore any sort.
    data = dsops.get_data(state.dataset, "numpy", wrap_single=True, ignore_sort=True)
    pixels = state.pixels
    if np.any(np.diff(pixels) < 0):
        order = np.argsort(pixels)
        pixels = pixels[order]
        data = {name: values[order] for name, values in data.items()}

    if format == "raw":
        return (pixels, data)
    elif format == "healsparse":
        maps = __make_healsparse_maps(pixels, data, state.nside, state.nside_lr)
    else:
        maps = __make_healpix_maps(pixels, data, state.nside)
    if len(maps) == 1:
        return next(iter(maps.values()))
    return maps


def with_resolution(state: HealpixMapState, nside: int) -> HealpixMapState:
    if nside == state.nside:
        return state
    elif nside > state.nside:
        raise ValueError(
            "You cannot change the resolution of a map to be higher than its original resolution!"
        )
    if not hp.isnsideok(nside):
        raise ValueError(f"New nside {nside} is invalid.")

    pixels, data = get_data(state, "raw")
    nside_ratio = state.nside // nside
    pixel_lores = pixels // (nside_ratio * nside_ratio)

    new_pixels, boundaries = np.unique(pixel_lores, return_index=True)
    counts = np.add.reduceat(np.ones_like(pixels), boundaries).astype(float)

    new_data = {
        name: (np.add.reduceat(values, boundaries) / counts).astype(np.float32)
        for name, values in data.items()
    }
    new_dataset = st.state_from_evaluated_data(
        new_data,
        state.header.with_parameters({"map_params/nside": nside}),
        {key: desc for key, desc in state.descriptions.items() if desc is not None},
    )
    sort_key = state.dataset.sort_key
    if sort_key is not None and sort_key[0] in new_data:
        new_dataset = replace(new_dataset, sort_key=sort_key)

    raw_pixels = None if len(new_pixels) == hp.nside2npix(nside) else new_pixels
    return HealpixMapState(new_dataset, raw_pixels)


def make_schema(state: HealpixMapState, path: str) -> Schema:
    # Rows are written in raw order to match the pixels in the header region.
    dataset = st.select(replace(state.dataset, sort_key=None), set(state.columns))
    ds_schema = st.make_schema(dataset, "/".join([path, "data"]))
    schema = make_map_schema(
        "/",
        FileEntry.HEALPIX_MAP,
        ds_schema.children,
        ds_schema.columns,
        ds_schema.attributes,
    )

    comm_world = get_comm_world()
    n_pixels = len(state) if comm_world is None else comm_world.allreduce(len(state))

    if n_pixels != hp.nside2npix(state.nside):
        new_header = state.header.with_region(state.region).with_parameter(
            "map_params/full_sky", False
        )
    else:
        new_header = state.header.with_region(FullSkyRegion())

    schema.children["header"] = new_header.dump()
    return schema


def cutouts(
    state: HealpixMapState,
    centers: SkyCoord,
    angular_size: float | u.Quantity,
    npix: int = 64,
) -> Generator[fits.HDUList, None, None]:
    if not isinstance(npix, int) or isinstance(npix, bool):
        raise TypeError("npix must be an integer")
    if npix <= 0:
        raise ValueError("npix must be positive")
    angular_size = validate_cutout_size(angular_size)

    pixels = get_included_pixels(centers, angular_size, state.nside)
    if len(centers) == 0:
        return

    if state.full_sky:
        rows = pixels
    else:
        pixels, _, rows = np.intersect1d(
            pixels, state.pixels, assume_unique=True, return_indices=True
        )

    if len(rows) == 0:
        data = {name: np.empty(0, dtype=np.float64) for name in state.columns}
    else:
        pixels, data = get_data(take_rows_unsorted(state, rows), "raw")

    for center in centers:
        region_pixels = make_skybox(center, angular_size).get_healpix_intersections(
            state.nside
        )
        region_pixels, index_to_include, _ = np.intersect1d(
            region_pixels, pixels, assume_unique=True, return_indices=True
        )
        region_data = {k: d[index_to_include] for k, d in data.items()}
        yield make_hdulist(
            region_pixels,
            region_data,
            state.nside,
            (center.ra.deg, center.dec.deg),
            angular_size,
            npix,
        )


def __make_healsparse_maps(
    pixels: np.ndarray,
    data: dict[str, np.ndarray],
    nside: int,
    nside_lr: int,
) -> dict[str, hsp.HealSparseMap]:
    sentinel = np.float32(hp.UNSEEN)

    # Build coverage map once, shared across all columns.
    cov_map = hsp.HealSparseCoverage.make_empty(nside_lr, nside)
    cov_pix = cov_map.cov_pixels(pixels)
    unique_cov_pix = np.unique(cov_pix)
    cov_map.initialize_pixels(unique_cov_pix)
    sparse_indices = pixels + cov_map[cov_pix]
    sparse_map_size = (len(unique_cov_pix) + 1) * cov_map.nfine_per_cov

    result = {}
    for name, values in data.items():
        sparse_map = np.full(sparse_map_size, sentinel, dtype=np.float32)
        sparse_map[sparse_indices] = values.astype(np.float32)
        result[name] = hsp.HealSparseMap(
            cov_map=cov_map,
            sparse_map=sparse_map,
            nside_sparse=nside,
            sentinel=sentinel,
        )
    return result


def __make_healpix_maps(
    pixels: np.ndarray, data: dict[str, np.ndarray], nside: int
) -> dict[str, np.ndarray]:
    npix = hp.nside2npix(nside)
    maps: dict[str, np.ndarray] = {}
    for name, values in data.items():
        maps[name] = np.full(npix, hp.UNSEEN, dtype=np.float32)
        maps[name][pixels] = values

    if len(pixels) == npix:
        return maps
    covered = np.zeros(npix, dtype=bool)
    covered[pixels] = True
    # Each column gets its own mask, so unmasking one leaves the rest.
    return {
        name: np.ma.MaskedArray(arr, mask=~covered, fill_value=hp.UNSEEN)
        for name, arr in maps.items()
    }


def __with_dataset(state: HealpixMapState, dataset: DatasetState) -> HealpixMapState:
    return replace(state, dataset=dataset)
