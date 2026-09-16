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
                "symbol": r"x",
                "scale": "linear",
                "min": None,
                "max": None,
            },
        }

    return params

def _set_plot_kwargs(plot_kwargs, **defaults):
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()

    for key, value in defaults.items():
        plot_kwargs.setdefault(key, value)

    return plot_kwargs

def _initialize_fig(ax):
    # create new figure if needed
    if ax is None:
        # if no axis is passed in, create a new figure
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    return fig, ax
  

def hist2d(ds, column_x, column_y, ax=None, plot_rank="all", plot_kwargs=None, **kwargs):

    fig=None

    x_params = _set_defaults(column_x)
    y_params = _set_defaults(column_y)

    h, x, y = statistics.hist2d(ds, column_x, column_y, **kwargs)

    if plot_rank=="all" or rank == plot_rank:
        fig, ax = _initialize_fig(ax)

        plot_kwargs = _set_plot_kwargs(plot_kwargs, norm=LogNorm(vmin=1))

        X, Y = np.meshgrid(x, y)
        ax.pcolormesh(X, Y, h.T, **plot_kwargs)

        ax.set(
            xlabel=x_params["plotting"]["label"],
            ylabel=y_params["plotting"]["label"],
            xscale=x_params["plotting"]["scale"],
            yscale=y_params["plotting"]["scale"],
        )

    return fig, ax


def hist1d(ds, column, differential=None, ax=None, plot_rank="all", plot_kwargs=None, **kwargs):  
    params = _set_defaults(column)

    counts, bin_edges = statistics.hist1d(ds, column, **kwargs)

    if differential == "linear":
        counts = counts / np.diff(bin_edges.value)
        ylabel = rf"$dN/d{params['plotting']['symbol']}$"

    elif differential == "log":
        counts = counts / np.diff(np.log10(bin_edges.value))
        ylabel = rf"$dN/d\log_{{10}}\left({params['plotting']['symbol']}\right)$"

    elif differential is None:
        ylabel = "N"

    else:
        raise ValueError(f"Invalid value for differential: {differential}")


    if plot_rank=="all" or rank == plot_rank:
        fig, ax = _initialize_fig(ax)

        plot_kwargs = _set_plot_kwargs(plot_kwargs, where="mid")        

        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])

        ax.step(bin_centers, counts, **plot_kwargs)

        ax.set(
            xlabel=params["plotting"]["label"],
            ylabel=ylabel,
            xscale=params["plotting"]["scale"],
            yscale="log"
        )

    return fig, ax

def binned_statistic(ds, column, statistic="mean", bin_by="sod_halo_mass", ax=None, plot_kwargs=None, plot_rank="all", **kwargs):
    x_params = _set_defaults(bin_by)
    y_params = _set_defaults(column)

    binned_stat, bin_edges = statistics.binned_statistic(ds, column, statistic=statistic, bin_by=bin_by, **kwargs)

    if plot_rank=="all" or rank == plot_rank:

        plot_kwargs = _set_plot_kwargs(plot_kwargs, linewidth=2)

        fig, ax = _initialize_fig(ax)

        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        ax.plot(bin_centers, binned_stat, **plot_kwargs)

        ax.set(
            xlabel=x_params["plotting"]["label"],
            ylabel=y_params["plotting"]["label"],
            xscale=x_params["plotting"]["scale"],
            yscale=y_params["plotting"]["scale"]
        )

    return fig, ax




def _scatter():
    return





def _stacked_profiles():
    # plot stacked profiles. If collection, plot all and color curves by given input column
    return

def plot_collection():
    # plot a binned_statistic for each simulation collection
    return

def plot_collage():
    # TODO: Allow a
    return