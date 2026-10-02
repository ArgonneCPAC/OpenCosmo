import astropy.units as u
import healpy as hp
import numpy as np
import pytest

import opencosmo as oc


@pytest.fixture
def lightcone_paths(test_data):
    return (
        test_data.lightcone.step(600).halo_properties,
        test_data.lightcone.step(601).halo_properties,
    )


def test_healpix_map_cone_center_and_overlap_policies(test_data):
    healpix_map = oc.open(test_data.healpix_map)
    region = oc.make_cone((45, -45), 2 * u.deg)
    vector = hp.ang2vec(45, -45, lonlat=True)

    center_pixels = hp.query_disc(
        healpix_map.nside,
        vector,
        np.deg2rad(2),
        inclusive=False,
        nest=healpix_map.ordering == "NESTED",
    )
    overlap_pixels = hp.query_disc(
        healpix_map.nside,
        vector,
        np.deg2rad(2),
        inclusive=True,
        nest=healpix_map.ordering == "NESTED",
    )

    center_result = healpix_map.bound(region, inclusive=False)
    overlap_result = healpix_map.bound(region, inclusive=True)

    assert np.array_equal(
        center_result.pixels, np.intersect1d(healpix_map.pixels, center_pixels)
    )
    assert np.array_equal(
        overlap_result.pixels, np.intersect1d(healpix_map.pixels, overlap_pixels)
    )
    assert set(center_result.pixels) < set(overlap_result.pixels)


@pytest.mark.parametrize(
    "region",
    [
        oc.make_cone((45, -45), 2 * u.deg),
        oc.make_skybox((43, -47), (47, -43)),
    ],
    ids=["cone", "skybox"],
)
@pytest.mark.filterwarnings("ignore:You're querying with a region")
def test_lightcone_bound_matches_catalog_and_map_children(
    test_data, lightcone_paths, region
):
    catalog = oc.open(*lightcone_paths)
    healpix_map = oc.open(test_data.healpix_map)
    lightcone = oc.open(*lightcone_paths, test_data.healpix_map)

    expected_catalog = catalog.bound(region)
    expected_map = healpix_map.bound(region)
    result = lightcone.bound(region)

    assert np.array_equal(
        result.select("fof_halo_tag").get_data("numpy"),
        expected_catalog.select("fof_halo_tag").get_data("numpy"),
    )
    assert np.array_equal(result.map.pixels, expected_map.pixels)
