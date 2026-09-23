from __future__ import annotations

from typing import TYPE_CHECKING

import healpy as hp
import numpy as np

from opencosmo._lib import spatial as spatlib

if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord


# Rings of coarse-pixel dilation applied around the pixels containing the
# cutout centers. The coarse grid is chosen so a cutout disc is no larger
# than a coarse pixel, so one ring is sufficient on geometric grounds; a
# second ring absorbs the irregularity of HEALPix neighborhoods near the
# base-pixel corners, where a pixel has only seven neighbors.
DILATION_RINGS = 2


def get_included_pixels(coordinates: SkyCoord, size: float, nside: int) -> np.ndarray:
    """
    Find every HEALPix pixel touched by a set of equally-sized square cutouts.

    Computes the union of required pixels up front so that map data only has
    to be fetched once, regardless of how many cutouts are requested. The work
    is fully vectorized: cutout centers are first deduplicated onto a coarse
    HEALPix grid, the resulting parent pixels are dilated to form a candidate
    superset, and the candidates are then filtered exactly against the cutout
    centers with a single nearest-neighbor query.

    Parameters
    ----------
    coordinates : SkyCoord
        Centers of the cutouts.
    size : float
        Angular width of each (square) cutout, in degrees.
    nside : int
        HEALPix resolution of the map the cutouts will be drawn from. Must be
        a positive power of two.

    Returns
    -------
    pixels : np.ndarray
        Sorted, unique pixel indices in nested ordering at ``nside``. These
        are purely geometric; intersect them with a map's own coverage to get
        the pixels that can actually be read.

    Raises
    ------
    ValueError
        If ``nside`` is not a positive power of two, or ``size`` is not
        positive.

    Notes
    -----
    The returned pixels cover the disc circumscribing each square cutout,
    padded by two pixels so that partially covered pixels and the bilinear
    interpolation stencil used during reprojection are included.
    """
    level = np.log2(nside)
    if not level.is_integer() or level < 0:
        raise ValueError("nside must be a positive power of two!")
    if size <= 0:
        raise ValueError("Cutout size must be positive!")

    centers = coordinates.reshape(-1)
    if len(centers) == 0:
        return np.empty(0, dtype=np.int64)

    # Disc circumscribing the square cutout, padded for partially covered
    # pixels and for the interpolation stencil used when reprojecting.
    radius = np.deg2rad(size) * np.sqrt(2.0) / 2.0 + 2.0 * hp.max_pixrad(nside)
    if radius >= np.pi:
        return np.arange(hp.nside2npix(nside), dtype=np.int64)

    # Deduplicate centers onto the finest grid whose pixels are still at
    # least as large as a cutout disc. Finer would fail to merge nearby
    # cutouts; coarser would inflate the candidate count per parent.
    level_c = 0
    while 2 ** (level_c + 1) <= nside and hp.max_pixrad(2 ** (level_c + 1)) >= radius:
        level_c += 1
    nside_c = 2**level_c

    parents = np.unique(
        hp.ang2pix(nside_c, centers.ra.deg, centers.dec.deg, lonlat=True, nest=True)
    )
    for _ in range(DILATION_RINGS):
        neighbours = hp.get_all_neighbours(nside_c, parents, nest=True).ravel()
        # get_all_neighbours returns -1 where a base-pixel corner leaves a
        # pixel with only seven neighbors.
        parents = np.union1d(parents, neighbours[neighbours >= 0])

    if nside_c == nside:
        candidates = parents
    else:
        # In nested ordering the children of a pixel are a contiguous run.
        ratio = (nside // nside_c) ** 2
        candidates = (parents[:, None] * ratio + np.arange(ratio)).ravel()

    # Keep the candidates whose nearest cutout center lies within the disc.
    # Angular separation is monotonic in chord length on the unit sphere, so
    # the cut can be applied directly in Cartesian space.
    center_vecs = (
        np.asarray(hp.ang2vec(centers.ra.deg, centers.dec.deg, lonlat=True))
        .reshape(-1, 3)
        .astype(np.float64)
    )  # Will revisit in the future
    candidate_vecs = (
        np.asarray(hp.pix2vec(nside, candidates, nest=True))
        .reshape(-1, 3)
        .astype(np.float64)
    )
    distances = spatlib.get_closest_distance_3d(
        center_vecs, candidate_vecs.astype(np.float64), 1
    )
    return np.sort(candidates[np.isfinite(distances)])
