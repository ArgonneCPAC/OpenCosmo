from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np

if TYPE_CHECKING:
    import opencosmo as oc


def make_radec_columns(dataset: oc.Lightcone):
    if "ra" in dataset.columns and "dec" in dataset.columns:
        return dataset
    elif "theta" in dataset.columns and "phi" in dataset.columns:
        return dataset.evaluate(
            radec_from_thetaphi, vectorize=True, insert=True, format="numpy"
        )
    else:
        warnings.warn(
            "Could not find coordinates in this catalog. Spatial queries will not be available"
        )


def radec_from_thetaphi(theta, phi):
    theta_deg = theta * 180 / np.pi
    phi_deg = phi * 180 / np.pi
    return {"ra": phi_deg * u.deg, "dec": (90.0 - theta_deg) * u.deg}
