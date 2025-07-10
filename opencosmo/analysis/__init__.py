# ruff: noqa
__all__ = []
yt_tools = [
    "create_yt_dataset",
    "ProjectionPlot",
    "SlicePlot",
    "ParticleProjectionPlot",
    "ProfilePlot",
    "PhasePlot",
    "visualize_halo",
    "halo_projection_array",
]

diffsky_tools = ["get_pop_mah"]

try:
    from .yt_utils import create_yt_dataset
    from .yt_viz import (
        ParticleProjectionPlot,
        PhasePlot,
        ProfilePlot,
        ProjectionPlot,
        SlicePlot,
        halo_projection_array,
        visualize_halo,
    )

    __all__.extend(yt_tools)

except ImportError:  # User has not installed yt tools
    pass

from .diffsky import get_pop_mah

__all__.extend(diffsky_tools)


"""
Right now, we have only have two analysis modules so we can handle them directly. In the 
future we will need to implement a more robust system that handles things automatically.
"""


def __getattr__(name):
    if name in yt_tools:
        raise ImportError(
            "You tried to import one of the OpenCosmo YT tools, but your python "
            "environment does not have the necessary dependencies. You can do install "
            "them with `pip install opencosmo[analysis]`"
        )
    raise ImportError(f"Cannot import name '{name}' from opencosmo.analysis")
