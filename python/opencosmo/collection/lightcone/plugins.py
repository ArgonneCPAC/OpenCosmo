from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

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


plugins = None
