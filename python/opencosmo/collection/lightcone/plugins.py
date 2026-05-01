from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np

import opencosmo as oc
from opencosmo.plugins.contexts import HookPoint
from opencosmo.plugins.hook import hook

if TYPE_CHECKING:
    from opencosmo import Lightcone
    from opencosmo.plugins.contexts import LightconeOpenCtx


@hook(HookPoint.LightconeOpen)
def _ensure_redshift_column(ctx: LightconeOpenCtx) -> LightconeOpenCtx:
    """Ensures a column called 'redshift' exists on every lightcone."""
    lightcone: Lightcone = ctx.lightcone
    if (
        "properties" not in lightcone.dtype
    ):  # Particles or profiles, redshift handled at structure collection level
        return ctx
    if "redshift" in lightcone.columns:
        return ctx
    elif "fof_halo_center_a" in lightcone.columns:
        z_col = 1 / oc.col("fof_halo_center_a") - 1
    elif "redshift_true" in lightcone.columns:
        z_col = oc.col("redshift_true")
    elif "zp" in lightcone.columns:
        z_col = oc.col("zp")
    else:
        raise ValueError(
            "Unable to find a redshift or scale factor column for this lightcone dataset"
        )
    return dataclasses.replace(
        ctx, lightcone=lightcone.with_new_columns(redshift=z_col)
    )


@hook(HookPoint.LightconeOpen)
def _make_radec_columns(ctx: LightconeOpenCtx):
    lightcone: Lightcone = ctx.lightcone
    if "ra" in lightcone.columns and "dec" in lightcone.columns:
        pass
    elif "theta" in lightcone.columns and "phi" in lightcone.columns:
        lightcone = lightcone.evaluate(
            radec_from_thetaphi, vectorize=True, insert=True, format="numpy"
        )
    elif "properties" in lightcone.dtype:
        warnings.warn(
            "Could not find coordinates in this catalog. Spatial queries will not be available"
        )

    return dataclasses.replace(ctx, lightcone=lightcone)


def radec_from_thetaphi(theta, phi):
    theta_deg = theta * 180 / np.pi
    phi_deg = phi * 180 / np.pi
    return {"ra": phi_deg * u.deg, "dec": (90.0 - theta_deg) * u.deg}
