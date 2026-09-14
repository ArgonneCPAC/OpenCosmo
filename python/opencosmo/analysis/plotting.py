import opencosmo.analysis.statistics as statistics
from opencosmo.analysis.default_plotting_params import default_params

from opencosmo.mpi import get_comm_world, get_mpi
MPI = get_mpi()
comm = get_comm_world()
rank = comm.Get_rank() if comm is not None else 0
ranks = comm.Get_size() if comm is not None else 1

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# all functions here should return a matplotlib figure
# all functions should also take an optional 'ax' input so that this can be easily integrated into collage

def _set_defaults(column):
    params = default_params.get(column)

    if params is None:
        params = {
            "filter_bad": None,
            "plotting": {
                "label": column,
                "scale": "log",
                "min": None,
                "max": None,
            },
        }

    return params
  

def hist2d(ds, column_x, column_y, ax=None, plot_rank="all", **kwargs):

    bins = kwargs.get("bins", 100)
    bin_spacing = kwargs.get("bin_spacing", "log")

    x_params = _set_defaults(column_x)
    y_params = _set_defaults(column_y)

    h, x, y = statistics.hist2d(ds, column_x, column_y, bins=bins, bin_spacing=bin_spacing)

    if plot_rank=="all" or rank == plot_rank:
        if ax is None:
            # if no axis is passed in, create a new figure
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        X, Y = np.meshgrid(x, y)
        ax.pcolormesh(X, Y, h.T, norm=LogNorm(vmin=1))

        ax.set(
            xlabel=x_params["plotting"]["label"],
            ylabel=y_params["plotting"]["label"],
            xscale=x_params["plotting"]["scale"],
            yscale=y_params["plotting"]["scale"],
        )

    return fig, ax

    





def _scatter():
    return





def _stacked_profiles():
    # plot stacked profiles. If collection, plot all and color curves by given input column
    return

def plot_collage():
    # TODO: Allow a
    return