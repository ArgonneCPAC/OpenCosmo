from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, TypeVar

import h5py
import numpy as np
from astropy import units as u
from healpy import ang2pix, nside2npix

from opencosmo.dataset import Dataset, state
from opencosmo.spatial.healpix import HealPixIndex
from opencosmo.spatial.octree import OctTreeIndex, get_octtree_index
from opencosmo.spatial.tree import Tree
from opencosmo.spatial.utils import combine_upwards

if TYPE_CHECKING:
    from opencosmo.header import OpenCosmoHeader

T = TypeVar("T")
GroupedColumnData = dict[str, dict[str, T]]
SpatialIndexData = dict[int, tuple[np.ndarray, int]]


def build_dataset_from_data(
    data: GroupedColumnData[np.ndarray],
    header: OpenCosmoHeader,
    spatial_index_data: SpatialIndexData | None,
    descriptions: GroupedColumnData[str] = {},
) -> Dataset:
    data_keys = set(data.keys())
    if data_keys != {"data"}:
        raise ValueError("Data must have exactly one `data` group")
    if descriptions and not set(descriptions.keys()).issubset(data.keys()):
        raise ValueError(
            "Descriptions should be organized into the same groups as the data!"
        )

    tree = None
    if isinstance(spatial_index_data, dict):
        spatial_index_columns = make_spatial_index(spatial_index_data)
        tree = Tree(HealPixIndex(), spatial_index_columns)
    data_group = data.pop("data")

    data_descriptions = descriptions.get("data", {})
    new_state = state.state_in_memory(
        data_group,
        header,
        header.file.unit_convention,
        {},
        data_descriptions,
        tree=tree,
    )
    return Dataset(new_state)


def build_dataset_from_evaluated_data(
    data: dict[str, np.ndarray | u.Quantity],
    header: OpenCosmoHeader,
    coordinate_names: list[str] | None = None,
    max_level: int | None = None,
) -> Dataset:
    """Build an in-memory dataset from results in the file's unit convention."""
    tree = None
    if coordinate_names is not None:
        if max_level is None:
            raise ValueError("Cannot build a spatial index without an index level")
        data, tree = build_tree_from_coordinates(
            data, coordinate_names, header, max_level
        )
    return Dataset(state.state_from_evaluated_data(data, header, tree=tree))


def build_tree_from_coordinates(
    data: dict[str, np.ndarray | u.Quantity],
    coordinate_names: list[str],
    header: OpenCosmoHeader,
    max_level: int,
) -> tuple[dict[str, np.ndarray | u.Quantity], Tree]:
    """Validate coordinate columns, spatially order data, and build its tree."""
    if max_level < 0:
        raise ValueError("Spatial index level must be non-negative")
    if header.file.is_lightcone:
        if coordinate_names != ["ra", "dec"]:
            raise ValueError("Lightcone coordinates must be ['ra', 'dec']")
        pixels = __get_healpix_pixels(data, max_level)
        index = HealPixIndex()
        region = None
    else:
        coordinate_names = __verify_snapshot_coordinates(
            data, coordinate_names, str(header.file.data_type)
        )
        box_size = header.with_units("scalefree").simulation["box_size"]
        pixels, box_size_value = __get_octree_pixels(
            data, coordinate_names, max_level, box_size
        )
        index = OctTreeIndex.from_box_size(int(box_size_value))
        from opencosmo.spatial.builders import make_box

        region = make_box((0, 0, 0), (box_size_value, box_size_value, box_size_value))

    order = np.argsort(pixels, kind="stable")
    sorted_data = {name: values[order] for name, values in data.items()}
    counts = np.bincount(
        pixels,
        minlength=__n_partitions(header.file.is_lightcone, max_level),
    )
    tree = Tree(
        index,
        make_spatial_index({max_level: (counts, index.subdivision_factor)}),
        region,
    )
    return sorted_data, tree


def __get_healpix_pixels(
    data: dict[str, np.ndarray | u.Quantity], max_level: int
) -> np.ndarray:
    ra = __to_float_values(data["ra"], u.deg)
    dec = __to_float_values(data["dec"], u.deg)
    if not np.all(np.isfinite(ra)) or not np.all(np.isfinite(dec)):
        raise ValueError("Coordinate columns must contain only finite values")
    if np.any(dec < -90) or np.any(dec > 90):
        raise ValueError("Declination coordinates must be between -90 and 90 degrees")
    return ang2pix(2**max_level, ra, dec, lonlat=True, nest=True)


def __get_octree_pixels(
    data: dict[str, np.ndarray | u.Quantity],
    coordinate_names: list[str],
    max_level: int,
    box_size: float | u.Quantity,
) -> tuple[np.ndarray, float]:
    coordinate_values: list[np.ndarray] = [data[name] for name in coordinate_names]
    if isinstance(box_size, u.Quantity):
        coordinate_values = [
            coordinate.to_value(box_size.unit)
            if isinstance(coordinate, u.Quantity)
            else coordinate
            for coordinate in coordinate_values
        ]
        box_size = float(box_size.value)
    else:
        coordinate_values = [
            __to_float_values(coordinate) for coordinate in coordinate_values
        ]
    coordinates = np.stack(coordinate_values, axis=1)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Coordinate columns must contain only finite values")
    if np.any(coordinates < 0) or np.any(coordinates >= box_size):
        raise ValueError(
            f"Snapshot coordinates must be greater than or equal to 0 and less than {box_size}"
        )
    blocks = (coordinates // (box_size / (2**max_level))).astype(int)
    pixels = np.fromiter(
        (get_octtree_index(tuple(block), max_level) for block in blocks),
        dtype=np.int64,
        count=len(blocks),
    )
    return pixels, float(box_size)


def __verify_snapshot_coordinates(
    data: dict[str, np.ndarray | u.Quantity], coordinate_names: list[str], dtype: str
) -> list[str]:
    from opencosmo.spatial.check import ALLOWED_COORDINATES_3D

    allowed_coordinates = ALLOWED_COORDINATES_3D.get(
        dtype, ALLOWED_COORDINATES_3D["default"]
    )
    expected_names = [
        base + dimension
        for base in allowed_coordinates.values()
        for dimension in ["x", "y", "z"]
    ]
    for base in allowed_coordinates.values():
        names = [base + dimension for dimension in ["x", "y", "z"]]
        if coordinate_names == names:
            return names
    raise ValueError(
        "Snapshot coordinates must be one of the recognized coordinate triplets: "
        f"{expected_names}"
    )


def __to_float_values(values: np.ndarray | u.Quantity, unit: u.Unit | None = None):
    if isinstance(values, u.Quantity):
        if unit is not None:
            return values.to_value(unit)
        return values.value
    return values


def __n_partitions(is_lightcone: bool, level: int) -> int:
    if is_lightcone:
        return nside2npix(2**level)
    return 8**level


def make_spatial_index(data: SpatialIndexData):
    """
    allowed input (for now)

    a single level > 0
    """
    if len(data) != 1:
        raise ValueError("Spatial index creation routines should have a single level")
    level = next(iter(data.keys()))
    size, fold_factor = data[level]
    if level < 0:
        raise ValueError(
            "Data for creating a spatial index must include a non-negative level"
        )
    name = uuid.uuid1()
    file = h5py.File(f"{name}.hdf5", "w", driver="core", backing_store=False)
    data = combine_upwards(size, fold_factor, level, file)
    output = {}
    for group in data.values():
        assert isinstance(group, h5py.Group)
        output[group.name[1:]] = group
        output.update({ds.name[1:]: ds for ds in group.values()})
    return output
