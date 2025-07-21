# ruff: noqa
from importlib import import_module

known_yt_tools = [
    "create_yt_dataset",
    "ProjectionPlot",
    "SlicePlot",
    "ParticleProjectionPlot",
    "ProfilePlot",
    "PhasePlot",
    "visualize_halo",
    "halo_projection_array",
]


"""
Right now, we have only have two analysis modules so we can handle them directly. In the 
future we will need to implement a more robust system that handles things automatically.
"""


def __getattr__(name):
    if name in known_yt_tools:
        yt_viz = import_module(".yt_viz", package="opencosmo.analysis")
        yt_utils = import_module(".yt_utils", package="opencosmo.analysis")
        try:
            return getattr(yt_viz, name)
        except AttributeError:
            pass
        try:
            return getattr(yt_utils, name)
        except AttributError:
            pass

        # except ImportError as ie:
        #    raise ImportError(
        #        "You tried to import one of the OpenCosmo YT tools, but your python "
        #        "environment does not have the necessary dependencies. You can do install "
        #        "them with `pip install opencosmo[analysis]`\n"
        #        f"{ie}"
        #    )
    raise ImportError(f"Cannot import name '{name}' from opencosmo.analysis")
