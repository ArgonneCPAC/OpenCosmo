from __future__ import annotations

from typing import TYPE_CHECKING

import opencosmo as oc
from opencosmo.plugins import PluginSpec, PluginType, register_plugin

if TYPE_CHECKING:
    from opencosmo import Lightcone


def with_redshift_column(dataset: Lightcone):
    """
    Ensures a column exists called "redshift" which contains the redshift of the objects
    in the lightcone.
    """
    if "redshift" in dataset.columns:
        return dataset

    elif "fof_halo_center_a" in dataset.columns:
        z_col = 1 / oc.col("fof_halo_center_a") - 1
        return dataset.with_new_columns(redshift=z_col)
    elif "redshift_true" in dataset.columns:
        z_col = oc.col("redshift_true")
        return dataset.with_new_columns(redshift=z_col)
    elif "zp" in dataset.columns:
        z_col = oc.col("zp")
        return dataset.with_new_columns(redshift=z_col)
    raise ValueError(
        "Unable to find a redshift or scale factor column for this lightcone dataset"
    )


register_plugin(
    PluginSpec(
        PluginType.LightconeLoad,
        lambda _: True,
        with_redshift_column,
    )
)

plugins = None
