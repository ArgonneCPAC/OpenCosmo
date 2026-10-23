from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from opencosmo.index import get_length, into_array
from opencosmo.spatial.builders import from_model
from opencosmo.spatial.region import FullSkyRegion, HealpixRegion

if TYPE_CHECKING:
    import numpy as np

    from opencosmo.dataset.state import DatasetState
    from opencosmo.header import OpenCosmoHeader


@dataclass(frozen=True, eq=False)
class HealpixMapState:
    """
    Backend state for a HealpixMap. Construct with ``from_dataset``.

    Each raw row of ``dataset``'s handler has a fixed pixel, so ``pixels`` is
    derived from the handler index and follows every row operation.
    """

    dataset: DatasetState
    # Pixel of each raw row of ``dataset``'s handler. None means the raw row
    # number is the pixel (a full-sky map).
    raw_pixels: np.ndarray | None = None

    def __post_init__(self):
        # Pixels are derived from the handler index, so it must cover every row.
        # An EmptyHandler keeps its rows in the cache and fails this check.
        index = self.dataset.raw_data_handler.index
        if get_length(index) != len(self.dataset):
            raise RuntimeError(
                "A healpix map must be backed by a handler that tracks its rows"
            )

    @property
    def pixels(self) -> np.ndarray:
        # Not ``dataset.raw_index``, which is in sort order. Map data is read
        # in raw-row order, so the pixels must be too.
        index = into_array(self.dataset.raw_data_handler.index)
        return index if self.raw_pixels is None else self.raw_pixels[index]

    @property
    def header(self) -> OpenCosmoHeader:
        return self.dataset.header

    @property
    def nside(self) -> int:
        return self.header.healpix_map["nside"]

    @property
    def nside_lr(self) -> int:
        return self.header.healpix_map["nside_lr"]

    @property
    def ordering(self) -> str:
        return self.header.healpix_map["ordering"]

    @property
    def z_range(self) -> tuple[float, float] | None:
        return self.header.healpix_map["z_range"]

    @property
    def region(self) -> HealpixRegion:
        return HealpixRegion(self.pixels, self.nside)

    @property
    def full_sky(self) -> bool:
        import healpy as hp

        # Rows and pixels are one-to-one, so this avoids materializing pixels.
        return len(self) == hp.nside2npix(self.nside)

    @property
    def columns(self) -> list[str]:
        return self.dataset.columns

    @property
    def descriptions(self) -> dict[str, str | None]:
        return self.dataset.descriptions

    def __len__(self) -> int:
        return len(self.dataset)


def from_dataset(dataset: DatasetState) -> HealpixMapState:
    """
    Build the state for a map read from a file. A partial-sky file stores the
    pixel of each row in its header region; a full-sky file stores one row per
    pixel, in pixel order.
    """
    header = dataset.header
    region = None if header.file.region is None else from_model(header.file.region)
    match region:
        case HealpixRegion():
            return HealpixMapState(dataset, region.pixels)
        case None | FullSkyRegion():
            return HealpixMapState(dataset)
        case _ if header.healpix_map["full_sky"]:
            return HealpixMapState(dataset)
        case _:
            raise ValueError(
                f"Healpix map files must have a healpix or full-sky region, got {type(region)}"
            )
