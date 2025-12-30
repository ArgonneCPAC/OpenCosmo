from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import yt  # type: ignore
from matplotlib.colors import LogNorm  # type: ignore
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar  # type: ignore
from unyt import unyt_quantity  # type: ignore
from yt.visualization.base_plot_types import get_multi_plot  # type: ignore

import opencosmo as oc
from opencosmo.analysis import create_yt_dataset

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from yt.visualization.plot_window import NormalPlot

# ruff: noqa: E501


def ParticleProjectionPlot(
    *args, **kwargs
) -> yt.AxisAlignedProjectionPlot | yt.OffAxisProjectionPlot:
    """
    Wrapper for `yt.ParticleProjectionPlot <https://yt-project.org/doc/reference/api/yt.visualization.plot_window.html#yt.visualization.plot_window.ParticleProjectionPlot>`_.

    Creates a 2D projection plot of particle-based data along a specified axis.

    Parameters
    ----------
    *args :
        Positional arguments passed directly to `yt.ParticleProjectionPlot`.
    **kwargs :
        Keyword arguments passed directly to `yt.ParticleProjectionPlot`.

    Returns
    -------
    yt.visualization.plot_window.ParticleProjectionPlot
        A ParticleProjectionPlot object containing the particle projection plot.
    """
    # mypy gets this wrong. ParticleProjectionPlot is basically a factory class
    return yt.ParticleProjectionPlot(*args, **kwargs)  # type: ignore


def ProjectionPlot(*args, **kwargs) -> NormalPlot:
    """
    Wrapper for `yt.ProjectionPlot <https://yt-project.org/doc/reference/api/yt.visualization.plot_window.html#yt.visualization.plot_window.ProjectionPlot>`_.

    Creates a 2D projection plot of particle-based data along a specified axis.
    Smoothing is applied to SPH particle data over the smoothing length

    Parameters
    ----------
    *args :
        Positional arguments passed directly to `yt.ProjectionPlot`.
    **kwargs :
        Keyword arguments passed directly to `yt.ProjectionPlot`.

    Returns
    -------
    yt.visualization.plot_window.ProjectionPlot
        A ProjectionPlot object containing the smoothed particle projection plot.
    """
    return yt.ProjectionPlot(*args, **kwargs)


def SlicePlot(*args, **kwargs) -> NormalPlot:
    """
    Wrapper for `yt.SlicePlot <https://yt-project.org/doc/reference/api/yt.visualization.plot_window.html#yt.visualization.plot_window.SlicePlot>`_.

    Creates a 2D slice plot of particle-based data along a specified axis.
    Smoothing is applied to SPH particle data over the smoothing length

    Parameters
    ----------
    *args :
        Positional arguments passed directly to `yt.SlicePlot`.
    **kwargs :
        Keyword arguments passed directly to `yt.SlicePlot`.

    Returns
    -------
    yt.visualization.plot_window.PlotWindow
        A PlotWindow object containing the particle slice plot.
    """
    return yt.SlicePlot(*args, **kwargs)


def ProfilePlot(*args, **kwargs) -> yt.ProfilePlot:
    """
    Wrapper for `yt.ProfilePlot <https://yt-project.org/doc/reference/api/yt.visualization.particle_plots.html#yt.visualization.particle_plots.ParticleProjectionPlot>`_.

    Creates a bin-averaged profile of a dependent variable
    as a function of one or more independent variables.

    Parameters
    ----------
    *args :
        Positional arguments passed directly to `yt.ProfilePlot`.
    **kwargs :
        Keyword arguments passed directly to `yt.ProfilePlot`.

    Returns
    -------
    yt.visualization.plot_window.PlotWindow
        A PlotWindow object containing the profile plot.
    """
    return yt.ProfilePlot(*args, **kwargs)


def PhasePlot(*args, **kwargs) -> yt.PhasePlot:
    """
    Wrapper for `yt.PhasePlot <https://yt-project.org/doc/reference/api/yt.visualization.profile_plotter.html#yt.visualization.profile_plotter.PhasePlot>`_.

    Creates a 2D histogram (phase plot) showing how one quantity varies as a function
    of two others, useful for visualizing thermodynamic or structural relationships
    (e.g., temperature vs. density colored by mass).

    Parameters
    ----------
    *args :
        Positional arguments passed directly to `yt.PhasePlot`.
    **kwargs :
        Keyword arguments passed directly to `yt.PhasePlot`.

    Returns
    -------
    yt.visualization.plot_window.PlotWindow
        A PlotWindow object containing the phase plot.
    """
    return yt.PhasePlot(*args, **kwargs)


def visualize_halo(
    halo_id: int,
    data: oc.StructureCollection,
    projection_axis: Optional[str] = "z",
    length_scale: Optional[str] = "top left",
    text_color: Optional[str] = "gray",
    width: Optional[float] = None,
) -> Figure:
    """
    Creates a figure showing particle projections of dark matter, stars, gas, and/or gas temperature
    for given halo. If any of the listed particle types are not present in the dataset, this will
    create a horizontal arrangement with only the particles/fields that are present. Otherwise,
    creates a 2x2-panel figure. Each panel is an 800x800 pixel array.

    To customize the arrangement of panels, fields, colormaps, etc., see
    :func:`halo_projection_array`.


    Parameters
    ----------
    halo_id : int
        Identifier of the halo to be visualized.
    data : opencosmo.StructureCollection
        OpenCosmo StructureCollection object containing both halo properties and particle data
        (e.g. output of ``opencosmo.open([haloproperties, sodbighaloparticles])``).
    projection_axis : str, optional
        Data is projected along this axis (``"x"``, ``"y"``, or ``"z"``).
        Overridden if ``params["projection_axes"]`` is provided
    length_scale : str or None, optional
        Optionally add a horizontal bar denoting length scale in Mpc.

        Options:
            - ``"top left"``: add to top left panel
            - ``"top right"``: add to top right panel
            - ``"bottom left"``: add to bottom left panel
            - ``"bottom right"``: add to bottom right panel
            - ``"all top"``: add to all panels on top row
            - ``"all bottom"``: add to all panels on bottom row
            - ``"all left"``: add to all panels on leftmost column
            - ``"all right"``: add to all panels on rightmost column
            - ``"all"``: add to all panels
            - ``None``: no length scale on any panel

    text_color : str, optional
        Set the color of all text annotations. Default is "gray"
    width : float, optional
        Width of each projection panel in units of R200 for the halo.
        If None, plots full subvolume around halo.

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib Figure object.
    """

    params: Dict[str, Any] = {
        "fields": [],
        "weight_fields": [],
        "zlims": [],
        "labels": [],
        "cmaps": [],
    }

    ptypes = [
        key.removesuffix("_particles")
        for key in data.keys()
        if key.endswith("_particles")
    ]

    any_supported = False

    if "dm" in ptypes:
        any_supported = True
        params["fields"].append(("dm", "particle_mass"))
        params["weight_fields"].append(None)
        params["zlims"].append(None)
        params["labels"].append("Dark Matter")
        params["cmaps"].append("gray")
    elif "gravity" in ptypes:
        any_supported = True
        # particle mass not stored for GO simulations because each particle has the same mass.
        # Use particle_ones for making images in this case instead
        params["fields"].append(("gravity", "particle_ones"))
        params["weight_fields"].append(None)
        params["zlims"].append(None)
        params["labels"].append("Dark Matter")
        params["cmaps"].append("gray")

    if "star" in ptypes:
        any_supported = True
        params["fields"].append(("star", "particle_mass"))
        params["weight_fields"].append(None)
        params["zlims"].append(None)
        params["labels"].append("Stars")
        params["cmaps"].append("bone")

    if "gas" in ptypes:
        any_supported = True
        params["fields"].append(("gas", "particle_mass"))
        params["weight_fields"].append(None)
        params["zlims"].append(None)
        params["labels"].append("Gas")
        params["cmaps"].append("viridis")
        # temperature field should always exist if gas
        # particles are present
        params["fields"].append(("gas", "temperature"))
        params["weight_fields"].append(("gas", "density"))
        params["zlims"].append((1e7, 1e8))
        params["labels"].append("Gas Temperature")
        params["cmaps"].append("inferno")

    if not any_supported:
        raise RuntimeError(
            "No compatible particle types present in dataset for this function. "
            'Possible options are "dm", "gravity", "star", and "gas".'
        )

    halo_ids: list[int] | tuple[list[int], list[int]]
    if len(params["fields"]) == 4:
        # if 4 fields, make a 2x2 figure
        halo_ids = ([halo_id, halo_id], [halo_id, halo_id])
        params = {key: (value[:2], value[2:]) for key, value in params.items()}

    else:
        # otherwise, do 1xN
        halo_ids = np.shape(params["fields"])[0] * [halo_id]
        params = {key: [value] for key, value in params.items()}

    return halo_projection_array(
        halo_ids,
        data,
        params=params,
        length_scale=length_scale,
        width=width,
        projection_axis=projection_axis,
        text_color=text_color,
    )


def halo_projection_array(
    halo_ids: int | list[int] | tuple[list[int], list[int]] | np.ndarray,
    data: oc.StructureCollection,
    field: Optional[Tuple[str, str]] = ("dm", "particle_mass"),
    weight_field: Optional[Tuple[str, str]] = None,
    projection_axis: Optional[str] = "z",
    cmap: Optional[str] = "gray",
    zlim: Optional[Tuple[float, float]] = None,
    params: Optional[Dict[str, Any]] = None,
    length_scale: Optional[str] = None,
    text_color: Optional[str] = "gray",
    width: Optional[float] = None,
) -> Figure:
    """
    Creates a multipanel figure of projections for different fields and/or halos.

    By default, creates an arrangement of dark matter particle projections with the
    same shape as `halo_ids`. Each panel is an 800x800 pixel array.

    Customizable — can change which fields are plotted for which halos, their order,
    weighting, etc., using `params`.

    **NOTE:** Dark matter particle masses often aren't stored for gravity-only simulations
    because the particles all have the same mass by construction. The particles are also
    labelled as "gravity" particles in this case instead of "dm" particles in the data.
    To project dark matter particles in gravity only, one can use the ``("gravity", "particle_ones")``
    field in place of ``("dm", "particle_mass")``. This will produce the same final image.

    Parameters
    ----------
    halo_ids : int or 2D array of int
        Unique ID of the halo(s) to be visualized. The shape of `halo_ids` sets the layout
        of the figure (e.g., if `halo_ids` is a 2x3 array, the outputted figure will be a 2x3
        array of projections). If `int`, a single panel is output while preserving formatting.
    data : opencosmo.StructureCollection
        OpenCosmo StructureCollection dataset containing both halo properties and particle data
        (e.g., output of ``opencosmo.open([haloproperties, sodbighaloparticles])``).
    field : tuple of str, optional
        Field to plot for all panels. Follows yt naming conventions (e.g., ``("dm", "particle_mass")``,
        ``("gas", "temperature")``). Overridden if ``params["fields"]`` is provided.
    weight_field : tuple of str, optional
        Field to weight by during projection. Follows yt naming conventions.
        Overridden if ``params["weight_fields"]`` is provided.
    projection_axis : str, optional
        Data is projected along this axis (``"x"``, ``"y"``, or ``"z"``).
        Overridden if ``params["projection_axes"]`` is provided
    cmap : str
        Matplotlib colormap to use for all panels. Overridden if ``params["cmaps"]`` is provided.
        See https://matplotlib.org/stable/gallery/color/colormap_reference.html for named colormaps.
    zlim : tuple of float, optional
        Colorbar limits for `field`. Overridden if ``params["zlims"]`` is provided.
    length_scale : str or None, optional
        Optionally add a horizontal bar denoting length scale in Mpc.

        Options:
            - ``"top left"``: add to top left panel
            - ``"top right"``: add to top right panel
            - ``"bottom left"``: add to bottom left panel
            - ``"bottom right"``: add to bottom right panel
            - ``"all top"``: add to all panels on top row
            - ``"all bottom"``: add to all panels on bottom row
            - ``"all left"``: add to all panels on leftmost column
            - ``"all right"``: add to all panels on rightmost column
            - ``"all"``: add to all panels
            - ``None``: no length scale shown

    params : dict, optional
        Dictionary of customization parameters for the projection panels. Overrides
        defaults. All values must be 2D arrays with the same shape as `halo_ids`.

        Keys may include:
            - ``"fields"``: 2D array of fields to plot (yt naming conventions)
            - ``"weight_fields"``: 2D array of projection weights (or None)
            - ``"projection_axes"``: 2D array of projection axes ("x", "y", or "z")
            - ``"zlims"``: 2D array of colorbar limits (log-scaled)
            - ``"labels"``: 2D array of panel labels (or None)
            - ``"cmaps"``: 2D array of Matplotlib colormaps for each panel
            - ``"widths"``: 2D array of widths in units of R200
    text_color : str, optional
        Set the color of all text annotations. Default is "gray"
    width : float, optional
        Width of each projection panel in units of R200 for the halo.
        Overridden if ``params["widths"]`` is provided.
        If None, plots full subvolume.

    Returns
    -------
    matplotlib.figure.Figure
        A Matplotlib Figure object.
    """

    halo_ids = np.atleast_2d(halo_ids)

    # determine shape of figure
    fig_shape = np.shape(halo_ids)

    # Default plotting parameters
    if weight_field is None:
        weight_field_ = np.full(fig_shape, None)
    else:
        weight_field_ = np.reshape(
            [weight_field for _ in range(np.prod(fig_shape))],
            (fig_shape[0], fig_shape[1], 2),
        )

    if zlim is None:
        zlim_ = np.full(fig_shape, None)
    else:
        zlim_ = np.reshape(
            [zlim for _ in range(np.prod(fig_shape))], (fig_shape[0], fig_shape[1], 2)
        )

    default_params = {
        "fields": (
            np.reshape(
                [field for _ in range(np.prod(fig_shape))],
                (fig_shape[0], fig_shape[1], 2),
            )
        ),
        "weight_fields": (weight_field_),
        "zlims": (zlim_),
        "projection_axes": (np.full(fig_shape, projection_axis)),
        "labels": (np.full(fig_shape, None)),
        "cmaps": (np.full(fig_shape, cmap)),
        "widths": (np.full(fig_shape, width)),
    }

    # Override defaults with user-supplied params (if any)
    params = params or {}

    fields = params.get("fields", default_params["fields"])
    weight_fields = params.get("weight_fields", default_params["weight_fields"])
    projection_axes = params.get("projection_axes", default_params["projection_axes"])
    zlims = params.get("zlims", default_params["zlims"])
    labels = params.get("labels", default_params["labels"])
    cmaps = params.get("cmaps", default_params["cmaps"])
    widths = params.get("widths", default_params["widths"])

    nrow, ncol = fig_shape

    ilen, jlen = None, None

    # define figure and axes
    fig, axes, cbars = get_multi_plot(fig_shape[1], fig_shape[0], cbar_padding=0)

    # are we plotting a single halo multiple times?
    halo_ids = np.array(halo_ids)
    halo_id_previous = np.inf

    for i in range(nrow):
        for j in range(ncol):
            halo_id = halo_ids[i][j]

            # retrieve halo particle info if new halo
            if (i == 0 and j == 0) or halo_id != halo_id_previous:
                # retrieve properties of halo
                data_id = data.filter(oc.col("unique_tag") == halo_id)
                halo_data = next(iter(data_id.objects()))

                # load particles into yt
                ds = create_yt_dataset(halo_data)

            halo_properties = halo_data["halo_properties"]

            Rh = unyt_quantity.from_astropy(halo_properties["sod_halo_radius"])

            field, weight_field, zlim, width = (
                tuple(fields[i][j]),
                weight_fields[i][j],
                zlims[i][j],
                widths[i][j],
            )

            if weight_field is not None:
                weight_field = tuple(weight_field)  # type: ignore
            if zlim is not None:
                zlim = tuple(zlim)  # type: ignore

            label = labels[i][j]

            proj = ParticleProjectionPlot(
                ds, projection_axes[i][j], field, weight_field=weight_field
            )

            proj.set_background_color(field, color="black")

            if width is not None:
                width_ = width
                proj.set_width(width_ * Rh)
            else:
                width_ = (max(ds.domain_width.to("Mpc")) / Rh).d  # type: ignore

            # fetch figure buffer (2D array of pixel values)
            # and re-plot on each panel with imshow
            frb = proj.frb

            ax = axes[i][j]

            if zlim is not None:
                zmin, zmax = zlim
            else:
                zmin, zmax = None, None

            ax.imshow(
                frb[field],
                origin="lower",
                cmap=cmaps[i][j],
                norm=LogNorm(vmin=zmin, vmax=zmax),
            )
            ax.set_facecolor("black")

            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            cbars[i].set_visible(False)

            if label is not None:
                # add panel label
                ax.text(
                    0.06,
                    0.94,
                    label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=12,
                    fontfamily="DejaVu Serif",
                    color=text_color,
                )

            if length_scale is not None:
                match length_scale:
                    case "top left":
                        ilen, jlen = 0, 0
                    case "top right":
                        ilen, jlen = 0, ncol - 1
                    case "bottom left":
                        ilen, jlen = nrow - 1, 0
                    case "bottom right":
                        ilen, jlen = nrow - 1, ncol - 1
                    case "all left":
                        ilen, jlen = i, 0
                    case "all right":
                        ilen, jlen = i, ncol - 1
                    case "all top":
                        ilen, jlen = 0, j
                    case "all bottom":
                        ilen, jlen = nrow - 1, j
                    case "all":
                        ilen, jlen = i, j

                if i == ilen and j == jlen:
                    # add length scale, assuming
                    # panel is 800 pixels wide
                    scalebar = AnchoredSizeBar(
                        ax.transData,
                        800 / (width_ * Rh.d),
                        "1 Mpc",
                        "lower right",
                        pad=0.4,
                        label_top=False,
                        sep=10,
                        color=text_color,
                        frameon=False,
                        size_vertical=1,
                    )
                    ax.add_artist(scalebar)

            halo_id_previous = halo_id

    return fig
