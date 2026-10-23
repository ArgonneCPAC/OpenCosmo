from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any

import astropy.cosmology.units as cu
import astropy.units as u
import numpy as np

import opencosmo as oc
from opencosmo.collection.lightcone import operations as lcops
from opencosmo.collection.lightcone import state as lcst
from opencosmo.column import norm_cols
from opencosmo.plugins.contexts import HookPoint
from opencosmo.plugins.hook import hook

if TYPE_CHECKING:
    from collections.abc import Callable

    from opencosmo.plugins.contexts import LightconeOpenCtx


@hook(
    HookPoint.LightconeOpen,
    when=lambda ctx: ctx.state.header.dtype in ["halo_properties", "galaxy_properties"],
)
def _ensure_coordinates(ctx: LightconeOpenCtx) -> LightconeOpenCtx:
    known_columns = set(ctx.state.columns)
    if known_columns.issuperset(("phi", "theta")) or known_columns.issuperset(
        ("ra", "dec")
    ):
        return ctx

    prefix = "fof_halo_center"
    if ctx.state.header.dtype == "galaxy_properties":
        prefix = "gal_center"

    coord_columns = {coord: f"{prefix}_{coord}" for coord in ["x", "y", "z"]}
    if not known_columns.issuperset(coord_columns.values()):
        raise ValueError("Unable to find coordinate columns for this lightcone dataset")
    chi = norm_cols(*list(coord_columns.values()))
    phi = oc.col(coord_columns["y"]).arctan2(oc.col(coord_columns["x"]))
    theta = (oc.col(coord_columns["z"]) / oc.col("chi")).arccos()
    new_state = lcops.with_new_columns(ctx.state, chi=chi, phi=phi, theta=theta)
    return dataclasses.replace(ctx, state=new_state)


@hook(HookPoint.LightconeOpen)
def _ensure_redshift_column(ctx: LightconeOpenCtx) -> LightconeOpenCtx:
    """Ensures a column called 'redshift' exists on every lightcone."""
    state = ctx.state
    dtype = state.header.dtype
    if (
        "particles" in dtype or "profiles" in dtype
    ):  # Particles or profiles, redshift handled at structure collection level
        return ctx
    if "redshift" in state.columns:
        return ctx
    elif "fof_halo_center_a" in state.columns:
        z_col = 1 / oc.col("fof_halo_center_a") - 1
    elif "redshift_true" in state.columns:
        z_col = oc.col("redshift_true")
    elif "zp" in state.columns:
        z_col = oc.col("zp")
    elif "chi" in state.columns:
        state = __evaluate(
            state,
            redshift_from_chi,
            cosmology=state.header.cosmology,
            vectorize=True,
        )
        return dataclasses.replace(ctx, state=state)

    else:
        raise ValueError(
            "Unable to find a redshift or scale factor column for this lightcone dataset"
        )
    return dataclasses.replace(ctx, state=lcops.with_new_columns(state, redshift=z_col))


@hook(HookPoint.LightconeOpen)
def _make_radec_columns(ctx: LightconeOpenCtx):
    state = ctx.state
    if "ra" in state.columns and "dec" in state.columns:
        pass
    elif "theta" in state.columns and "phi" in state.columns:
        state = __evaluate(state, radec_from_thetaphi, vectorize=True, format="numpy")
    elif "properties" in state.header.dtype:
        warnings.warn(
            "Could not find coordinates in this catalog. Spatial queries will not be available"
        )

    return dataclasses.replace(ctx, state=state)


def __evaluate(
    state: lcst.LightconeState, func: Callable, **kwargs: Any
) -> lcst.LightconeState:
    result = lcops.evaluate(state, func, insert=True, **kwargs)
    assert isinstance(result, lcst.LightconeState)
    return result


def radec_from_thetaphi(theta, phi):
    theta_deg = theta * 180 / np.pi
    phi_deg = phi * 180 / np.pi
    return {"ra": phi_deg * u.deg, "dec": (90.0 - theta_deg) * u.deg}


def redshift_from_chi(chi: u.Quantity, cosmology):
    distance = chi.to(u.Mpc, cu.with_H0(cosmology.H0))

    if len(distance) == 0:
        # astropy's z_at_value (used by the redshift-distance equivalency below)
        # opens an np.nditer over the input, which raises "Iteration of
        # zero-sized operands is not enabled" on empty input. Surplus MPI ranks
        # hold zero-length datasets, so short-circuit with an empty result of the
        # right unit rather than invoking the solver.
        return {"redshift": distance.value * cu.redshift}

    z_max = distance.max().to(
        cu.redshift, cu.redshift_distance(cosmology, kind="comoving")
    )
    z_min = distance.min().to(
        cu.redshift, cu.redshift_distance(cosmology, kind="comoving")
    )
    z_grid = np.linspace(z_min.value, z_max.value, 4096)
    chi_grid = cosmology.comoving_distance(z_grid).to(u.Mpc)
    z = np.interp(distance.to_value(u.Mpc), chi_grid.to_value(u.Mpc), z_grid)
    return {"redshift": z * cu.redshift}
