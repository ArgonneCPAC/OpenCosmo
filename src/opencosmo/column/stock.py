import astropy.units as u

from .column import col


def add_mag_cols(*magnitude_names):
    if len(magnitude_names) < 2:
        raise ValueError("Expected at least two magnitudes to add together!")
    fluxes = map(
        lambda colname: (-0.4 * col(colname)).exp10(expected_unit_container=u.MagUnit),
        magnitude_names,
    )
    total_flux = next(fluxes)
    for flux in fluxes:
        total_flux += flux

    return -2.5 * total_flux.log10(u.MagUnit)
