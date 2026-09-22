"""
Matplotlib front ends for the functions in
:py:mod:`opencosmo.analysis.statistics`.

Each plotting function mirrors the signature of the statistics function it
wraps, forwarding any extra keyword arguments (``bins``, ``bin_spacing``,
``mode``, and so on) straight through. On top of that it adds three things:

* An optional ``ax``, so a panel can be dropped into an existing figure. If
  none is given, a new figure is created.
* Axis labels and log/linear scales filled in automatically from
  :py:data:`default_params <opencosmo.analysis.default_plotting_params.default_params>`.
  Columns with no entry fall back to using the column name as its own label on
  a linear axis.
* A ``plot_rank`` switch controlling which MPI ranks draw. The underlying
  statistic is always computed collectively on every rank -- ``plot_rank``
  only decides who renders it.

Every function returns a ``(fig, ax)`` pair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

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

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from opencosmo import Dataset, StructureCollection
    from opencosmo.analysis.statistics import Statistic

    Differential = Literal["linear", "log"]
    PlotRank = int | Literal["all"]

# all functions here should return a matplotlib figure
# all functions should also take an optional 'ax' input so that this can be easily integrated into collage

def _set_defaults(column: str) -> dict[str, Any]:
    """
    Look up the plotting defaults for a column.

    Parameters
    ----------
    column : str
        The column to look up.

    Returns
    -------
    params : dict
        The column's entry in
        :py:data:`default_params <opencosmo.analysis.default_plotting_params.default_params>`.
        For a column with no entry, a fallback is synthesized that labels the
        axis with the raw column name, uses "x" as its math symbol, and puts it
        on a linear scale.
    """
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

def _set_plot_kwargs(
    plot_kwargs: dict[str, Any] | None, **defaults: Any
) -> dict[str, Any]:
    """
    Merge caller-supplied matplotlib kwargs over a set of defaults.

    Anything the caller passes wins, so a default is only applied when the key
    is absent. The caller's dictionary is copied rather than mutated.

    Parameters
    ----------
    plot_kwargs : dict or None
        The kwargs supplied by the caller. :code:`None` is treated as empty.
    **defaults
        The defaults to fall back on.

    Returns
    -------
    plot_kwargs : dict
        The merged kwargs, ready to hand to a matplotlib artist.
    """
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()

    for key, value in defaults.items():
        plot_kwargs.setdefault(key, value)

    return plot_kwargs

def _initialize_fig(ax: Axes | None) -> tuple[Figure, Axes]:
    """
    Resolve the figure to draw into, creating one if needed.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        An existing axis to draw into. If :code:`None`, a new single-panel
        figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure owning ``ax``.
    ax : matplotlib.axes.Axes
        The axis to draw into.
    """
    # create new figure if needed
    if ax is None:
        # if no axis is passed in, create a new figure
        fig, ax = plt.subplots()
    else:
        # .figure is typed Figure | SubFigure; an Axes drawn into by these
        # helpers always belongs to a full Figure.
        fig = cast("Figure", ax.figure)

    return fig, ax
  

def hist2d(
    ds: Dataset | StructureCollection,
    column_x: str,
    column_y: str,
    ax: Axes | None = None,
    plot_rank: PlotRank = "all",
    plot_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[Figure | None, Axes | None]:
    """
    Plot a two-dimensional histogram as a pcolormesh.

    Wraps :py:func:`opencosmo.analysis.statistics.hist2d` and renders the
    result with a logarithmic color normalization.

    .. code-block:: python

        import matplotlib.pyplot as plt
        import opencosmo as oc
        from opencosmo.analysis import plotting

        ds = oc.open("haloproperties.hdf5")

        fig, ax = plotting.hist2d(
            ds,
            "sod_halo_mass",
            "sod_halo_cdelta",
            bins=80,
            bin_spacing=("log", "linear"),
            plot_kwargs={"cmap": "magma"},
        )
        fig.savefig("concentration_mass.png")

    Parameters
    ----------
    ds : opencosmo.Dataset
        The data to plot.
    column_x, column_y : str
        The columns to histogram along the horizontal and vertical axes.
    ax : matplotlib.axes.Axes, optional
        An existing axis to draw into. If omitted, a new figure is created.
        Pass one of these to build a multi-panel figure.
    plot_rank : str or int, default = "all"
        Which MPI rank draws the plot. ``"all"`` draws on every rank; an
        integer draws only on that rank, which is usually what you want when
        writing a single image to disk (especially when working with a large number of ranks). 
        The histogram itself is computed collectively regardless.
    plot_kwargs : dict, optional
        Passed to :py:meth:`~matplotlib.axes.Axes.pcolormesh`. Defaults to
        :code:`{"norm": LogNorm(vmin=1)}`, which clips empty bins; anything you
        supply here takes precedence.
    **kwargs
        Forwarded to :py:func:`opencosmo.analysis.statistics.hist2d`, e.g.
        ``bins``, ``bin_spacing``, ``mode``.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure that was drawn into, or :code:`None` on ranks that did not
        draw.
    ax : matplotlib.axes.Axes or None
        The axis that was drawn into, or :code:`None` on ranks that did not draw.
    """

    fig=None

    x_params = _set_defaults(column_x)
    y_params = _set_defaults(column_y)

    # set default bin spacing to axis scale ("log" or "linear") if bins aren't explicitly defined
    if isinstance(kwargs.get("bins", 100), int) and "bin_spacing" not in kwargs:
        kwargs["bin_spacing"] = (
            x_params["plotting"]["scale"],
            y_params["plotting"]["scale"]
        )

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


def hist1d(
    ds: Dataset | StructureCollection,
    column: str,
    differential: Differential | None = None,
    yscale: str = "log"
    ax: Axes | None = None,
    plot_rank: PlotRank = "all",
    plot_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[Figure | None, Axes | None]:
    r"""
    Plot a one-dimensional histogram as a step curve.

    Wraps :py:func:`opencosmo.analysis.statistics.hist1d`.

    The ``differential`` argument divides the raw counts by the bin widths,
    turning the histogram into a density.

    .. code-block:: python

        import opencosmo as oc
        from opencosmo.analysis import plotting

        ds = oc.open("haloproperties.hdf5")

        # dN / dlog10(M200c), drawn only on rank 0
        fig, ax = plotting.hist1d(
            ds,
            "sod_halo_mass",
            differential="log",
            bins=30,
            plot_rank=0,
        )

    Parameters
    ----------
    ds : opencosmo.Dataset
        The data to plot.
    column : str
        The column to histogram.
    differential : str, optional
        How to normalize the counts by bin width.

        * :code:`None` (default) -- plot raw counts, labeled :math:`N`.
        * ``"linear"`` -- divide by :math:`\Delta x`, giving :math:`dN/dx`.
        * ``"log"`` -- divide by :math:`\Delta \log_{10} x`, giving
          :math:`dN/d\log_{10}(x)`.
    ax : matplotlib.axes.Axes, optional
        An existing axis to draw into. If omitted, a new figure is created.
    plot_rank : str or int, default = "all"
        Which MPI rank draws the plot. ``"all"`` draws on every rank; an
        integer draws only on that rank. The histogram is computed
        collectively on all ranks regardless.
    plot_kwargs : dict, optional
        Passed to :py:meth:`~matplotlib.axes.Axes.step`. Defaults to
        :code:`{"where": "mid"}` to draw the curve at the bin centers;
        anything you supply here takes precedence.
    **kwargs
        Forwarded to :py:func:`opencosmo.analysis.statistics.hist1d`, e.g.
        ``bins``, ``bin_spacing``, ``mode``.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure that was drawn into, or :code:`None` on ranks that did not
        draw.
    ax : matplotlib.axes.Axes or None
        The axis that was drawn into, or :code:`None` on ranks that did not draw.

    Raises
    ------
    ValueError
        If ``differential`` is not :code:`None`, "linear", or "log".
    """
    params = _set_defaults(column)

    # set default bin spacing to axis scale ("log" or "linear") if bins aren't explicitly defined
    if isinstance(kwargs.get("bins", 100), int) and "bin_spacing" not in kwargs:
        kwargs["bin_spacing"] = params["plotting"]["scale"]

    counts, bin_edges = statistics.hist1d(ds, column, **kwargs)

    # the differential forms need unit-ful edges
    edges_with_units: Any = bin_edges

    if differential == "linear":
        counts = counts / np.diff(edges_with_units.value)
        ylabel = rf"$dN/d{params['plotting']['symbol']}$"

    elif differential == "log":
        counts = counts / np.diff(np.log10(edges_with_units.value))
        ylabel = rf"$dN/d\log_{{10}}\left({params['plotting']['symbol']}\right)$"

    elif differential is None:
        ylabel = "N"

    else:
        raise ValueError(f"Invalid value for differential: {differential}")


    fig = None
    if plot_rank=="all" or rank == plot_rank:
        fig, ax = _initialize_fig(ax)

        plot_kwargs = _set_plot_kwargs(plot_kwargs, where="mid")        

        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])

        ax.step(bin_centers, counts, **plot_kwargs)

        ax.set(
            xlabel=params["plotting"]["label"],
            ylabel=ylabel,
            xscale=params["plotting"]["scale"],
            yscale=yscale
        )

    return fig, ax

def binned_statistic(
    ds: Dataset | StructureCollection,
    column: str,
    statistic: Statistic = "mean",
    bin_by: str = "sod_halo_mass",
    ax: Axes | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    plot_rank: PlotRank = "all",
    **kwargs: Any,
) -> tuple[Figure | None, Axes | None]:
    r"""
    Plot a statistic of ``column`` against ``bin_by`` as a line.

    Wraps :py:func:`opencosmo.analysis.statistics.binned_statistic` and plots
    the result at the bin centers.

    Because the function accepts an ``ax`` and returns the one it drew into,
    calling this function repeatedly allows one to overplot several curves on one panel.

    .. code-block:: python

        import matplotlib.pyplot as plt
        import opencosmo as oc
        from opencosmo.analysis import plotting

        ds = oc.open("haloproperties.hdf5")

        # median concentration-mass relation with a 16th-84th percentile band
        fig, ax = plotting.binned_statistic(
            ds, "sod_halo_cdelta", statistic="median"
        )
        for q, style in ((0.16, ":"), (0.84, ":")):
            plotting.binned_statistic(
                ds,
                "sod_halo_cdelta",
                statistic="quantile",
                q=q,
                ax=ax,
                plot_kwargs={"linestyle": style, "color": "k"},
            )

    Parameters
    ----------
    ds : opencosmo.Dataset or opencosmo.StructureCollection
        The data to plot.
    column : str
        The column to compute the statistic of. Sets the vertical axis.
    statistic : str or Callable, default = "mean"
        The reduction to apply within each bin. See
        :py:func:`opencosmo.analysis.statistics.binned_statistic` for the full
        list, including the "geometric\_" variants.
    bin_by : str, default = "sod_halo_mass"
        The column whose values define the bins. Sets the horizontal axis.
    ax : matplotlib.axes.Axes, optional
        An existing axis to draw into. If omitted, a new figure is created.
    plot_kwargs : dict, optional
        Passed to :py:meth:`~matplotlib.axes.Axes.plot`. Defaults to
        :code:`linewidth=2`; anything you supply here takes precedence.
    plot_rank : str or int, default = "all"
        Which MPI rank draws the plot. ``"all"`` draws on every rank; an
        integer draws only on that rank. The statistic is computed
        collectively on all ranks regardless.
    **kwargs
        Forwarded to
        :py:func:`opencosmo.analysis.statistics.binned_statistic`. This covers
        both the binning arguments (``bins``, ``dataset``, ``mode``) and any
        arguments consumed by the statistic itself, such as ``q`` for
        :code:`statistic="quantile"`.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure that was drawn into, or :code:`None` on ranks that did not
        draw.
    ax : matplotlib.axes.Axes or None
        The axis that was drawn into, or :code:`None` on ranks that did not draw.
    """
    x_params = _set_defaults(bin_by)
    y_params = _set_defaults(column)

    binned_stat, bin_edges = statistics.binned_statistic(ds, column, statistic=statistic, bin_by=bin_by, **kwargs)

    fig=None
    if plot_rank=="all" or rank == plot_rank:

        plot_kwargs = _set_plot_kwargs(plot_kwargs, linewidth=2)

        fig, ax = _initialize_fig(ax)

        edges = np.asarray(bin_edges)
        bin_centers = 0.5*(edges[1:] + edges[:-1])
        ax.plot(bin_centers, binned_stat, **plot_kwargs)

        ax.set(
            xlabel=x_params["plotting"]["label"],
            ylabel=y_params["plotting"]["label"],
            xscale=x_params["plotting"]["scale"],
            yscale=y_params["plotting"]["scale"]
        )

    return fig, ax