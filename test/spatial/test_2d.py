import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

import opencosmo as oc

# ---------------------------------------------------------------------------
# Cone search (bound with ConeRegion)
# ---------------------------------------------------------------------------


def test_cone_search(haloproperties_600_path):
    ds = oc.open(haloproperties_600_path)
    raw_data = ds.get_data()

    center = (45 * u.deg, -45 * u.deg)
    radius = 2 * u.deg
    center_coord = SkyCoord(*center)

    raw_data_coords = SkyCoord(
        raw_data["phi"], np.pi / 2 - raw_data["theta"], unit="rad"
    )
    raw_data_seps = center_coord.separation(raw_data_coords)
    n_raw = np.sum(raw_data_seps < radius)

    region = oc.make_cone(center, radius)
    data = ds.bound(region).get_data()
    ra = data["phi"]
    dec = np.pi / 2 - data["theta"]

    coordinates = SkyCoord(ra, dec, unit="radian")
    seps = center_coord.separation(coordinates)
    seps = seps.to(u.degree)
    assert all(seps < radius)
    assert len(data) == n_raw


def test_cone_search_chain_failure(haloproperties_600_path):
    ds = oc.open(haloproperties_600_path)

    center1 = (45 * u.deg, -45 * u.deg)
    center2 = (45 * u.deg, 45 * u.deg)
    radius = 2 * u.deg

    region1 = oc.make_cone(center1, radius)
    region2 = oc.make_cone(center2, radius)
    ds = ds.bound(region1)
    ds = ds.bound(region2)
    assert len(ds) == 0


def test_cone_search_chain(haloproperties_600_path):
    ds = oc.open(haloproperties_600_path)
    raw_data = ds.get_data()

    center = (45 * u.deg, -45 * u.deg)
    center_coord = SkyCoord(*center)
    radius1 = 2 * u.deg
    radius2 = 1 * u.deg

    region1 = oc.make_cone(center, radius1)
    region2 = oc.make_cone(center, radius2)
    ds = ds.bound(region1)
    ds = ds.bound(region2)

    raw_data_coords = SkyCoord(
        raw_data["phi"], np.pi / 2 - raw_data["theta"], unit="rad"
    )
    raw_data_seps = center_coord.separation(raw_data_coords)
    n_raw = np.sum(raw_data_seps < radius2)

    assert n_raw == len(ds)


def test_cone_search_write(haloproperties_600_path, tmp_path):
    ds = oc.open(haloproperties_600_path)

    center = (45, -45)
    radius = 4 * u.deg

    region = oc.make_cone(center, radius)
    ds = ds.bound(region)

    oc.write(tmp_path / "lightcone_test.hdf5", ds)
    new_ds = oc.open(tmp_path / "lightcone_test.hdf5")

    radius2 = 2 * u.deg
    region2 = oc.make_cone(center, radius2)
    ds = ds.bound(region2)
    new_ds = new_ds.bound(region2)

    assert set(ds.get_data()["fof_halo_tag"]) == set(new_ds.get_data()["fof_halo_tag"])


def test_cone_search_write_fail(haloproperties_600_path, tmp_path):
    ds = oc.open(haloproperties_600_path)

    center = (45 * u.deg, -45 * u.deg)
    radius = 2 * u.deg

    region = oc.make_cone(center, radius)
    ds = ds.bound(region)

    oc.write(tmp_path / "lightcone_test.hdf5", ds)
    new_ds = oc.open(tmp_path / "lightcone_test.hdf5")

    center2 = (45 * u.deg, 45 * u.deg)
    region2 = oc.make_cone(center2, radius)
    new_ds = new_ds.bound(region2)
    assert len(new_ds) == 0


def test_cone_search_collection(haloproperties_600_path, haloproperties_601_path):
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    raw_data = ds.get_data()

    center = (45 * u.deg, -45 * u.deg)
    radius = 2 * u.deg
    center_coord = SkyCoord(*center)

    raw_data_coords = SkyCoord(
        raw_data["phi"], np.pi / 2 - raw_data["theta"], unit="rad"
    )
    raw_data_seps = center_coord.separation(raw_data_coords)
    n_raw = np.sum(raw_data_seps < radius)

    region = oc.make_cone(center, radius)
    data = ds.bound(region).get_data()
    ra = data["phi"]
    dec = np.pi / 2 - data["theta"]

    coordinates = SkyCoord(ra, dec, unit="radian")
    seps = center_coord.separation(coordinates)
    seps = seps.to(u.degree)
    assert all(seps < radius)
    assert len(data) == n_raw


# ---------------------------------------------------------------------------
# Box search (box_search / bound with SkyboxRegion)
# ---------------------------------------------------------------------------

# Data is concentrated around RA ≈ 45°, Dec ≈ -45°, as confirmed by the
# cone_search tests above that use center (45°, -45°) and find data there.
# The disjoint area (45°, 45°) returns zero results.

_BOX_RA_MIN = 43.0
_BOX_RA_MAX = 47.0
_BOX_DEC_MIN = -47.0
_BOX_DEC_MAX = -43.0


def _raw_box_mask(raw_data, ra_min, ra_max, dec_min, dec_max):
    """Return boolean mask for rows strictly inside the RA/Dec box."""
    raw_ra = np.rad2deg(raw_data["phi"])
    raw_dec = np.rad2deg(np.pi / 2 - raw_data["theta"])
    return (
        (raw_ra > ra_min)
        & (raw_ra < ra_max)
        & (raw_dec > dec_min)
        & (raw_dec < dec_max)
    )


def test_box_search(haloproperties_600_path):
    """box_search returns exactly the rows within the RA/Dec box."""
    ds = oc.open(haloproperties_600_path)
    raw_data = ds.get_data()

    p1 = SkyCoord(_BOX_RA_MIN * u.deg, _BOX_DEC_MIN * u.deg)
    p2 = SkyCoord(_BOX_RA_MAX * u.deg, _BOX_DEC_MAX * u.deg)

    n_expected = np.sum(
        _raw_box_mask(raw_data, _BOX_RA_MIN, _BOX_RA_MAX, _BOX_DEC_MIN, _BOX_DEC_MAX)
    )
    assert n_expected > 0, "Test setup error: no raw data in target box"

    result = ds.box_search(p1, p2)
    data = result.get_data()

    ra = np.rad2deg(data["phi"])
    dec = np.rad2deg(np.pi / 2 - data["theta"])

    assert np.all((ra > _BOX_RA_MIN) & (ra < _BOX_RA_MAX))
    assert np.all((dec > _BOX_DEC_MIN) & (dec < _BOX_DEC_MAX))
    assert len(data) == n_expected


def test_box_search_tuple_args(haloproperties_600_path):
    """box_search accepts plain (RA, Dec) tuples as well as SkyCoord."""
    ds = oc.open(haloproperties_600_path)
    raw_data = ds.get_data()

    n_expected = np.sum(
        _raw_box_mask(raw_data, _BOX_RA_MIN, _BOX_RA_MAX, _BOX_DEC_MIN, _BOX_DEC_MAX)
    )

    result = ds.box_search((_BOX_RA_MIN, _BOX_DEC_MIN), (_BOX_RA_MAX, _BOX_DEC_MAX))
    assert len(result) == n_expected


def test_box_search_empty_region(haloproperties_600_path):
    """box_search in an area with no data returns an empty dataset."""
    ds = oc.open(haloproperties_600_path)
    p1 = SkyCoord(43 * u.deg, 43 * u.deg)
    p2 = SkyCoord(47 * u.deg, 47 * u.deg)
    result = ds.box_search(p1, p2)
    assert len(result) == 0


def test_box_search_chain(haloproperties_600_path):
    """Chaining two box_search calls applies the intersection of both boxes."""
    ds = oc.open(haloproperties_600_path)
    raw_data = ds.get_data()

    # Outer box is wider; inner box is the reference box
    outer_ra_min, outer_ra_max = 42.0, 48.0
    outer_dec_min, outer_dec_max = -48.0, -42.0

    p1_outer = SkyCoord(outer_ra_min * u.deg, outer_dec_min * u.deg)
    p2_outer = SkyCoord(outer_ra_max * u.deg, outer_dec_max * u.deg)
    p1_inner = SkyCoord(_BOX_RA_MIN * u.deg, _BOX_DEC_MIN * u.deg)
    p2_inner = SkyCoord(_BOX_RA_MAX * u.deg, _BOX_DEC_MAX * u.deg)

    n_expected = np.sum(
        _raw_box_mask(raw_data, _BOX_RA_MIN, _BOX_RA_MAX, _BOX_DEC_MIN, _BOX_DEC_MAX)
    )

    result = ds.box_search(p1_outer, p2_outer).box_search(p1_inner, p2_inner)
    assert len(result) == n_expected


def test_box_search_chain_failure(haloproperties_600_path):
    """Chaining two disjoint box_search calls produces an empty dataset."""
    ds = oc.open(haloproperties_600_path)

    p1 = SkyCoord(_BOX_RA_MIN * u.deg, _BOX_DEC_MIN * u.deg)
    p2 = SkyCoord(_BOX_RA_MAX * u.deg, _BOX_DEC_MAX * u.deg)
    # Disjoint box — same RA range but Dec on the opposite side of the sky
    p3 = SkyCoord(_BOX_RA_MIN * u.deg, 43 * u.deg)
    p4 = SkyCoord(_BOX_RA_MAX * u.deg, 47 * u.deg)

    result = ds.box_search(p1, p2).box_search(p3, p4)
    assert len(result) == 0


def test_box_search_collection(haloproperties_600_path, haloproperties_601_path):
    """box_search on a multi-step lightcone collection returns the correct rows."""
    ds = oc.open(haloproperties_600_path, haloproperties_601_path)
    raw_data = ds.get_data()

    p1 = SkyCoord(_BOX_RA_MIN * u.deg, _BOX_DEC_MIN * u.deg)
    p2 = SkyCoord(_BOX_RA_MAX * u.deg, _BOX_DEC_MAX * u.deg)

    n_expected = np.sum(
        _raw_box_mask(raw_data, _BOX_RA_MIN, _BOX_RA_MAX, _BOX_DEC_MIN, _BOX_DEC_MAX)
    )
    assert n_expected > 0, "Test setup error: no raw data in target box"

    result = ds.box_search(p1, p2)
    data = result.get_data()

    ra = np.rad2deg(data["phi"])
    dec = np.rad2deg(np.pi / 2 - data["theta"])

    assert np.all((ra > _BOX_RA_MIN) & (ra < _BOX_RA_MAX))
    assert np.all((dec > _BOX_DEC_MIN) & (dec < _BOX_DEC_MAX))
    assert len(data) == n_expected
