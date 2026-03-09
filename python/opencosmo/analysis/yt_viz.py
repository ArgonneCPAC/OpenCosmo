from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import yt  # type: ignore
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # type: ignore
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar  # type: ignore
from unyt import unyt_quantity  # type: ignore
from yt.visualization.base_plot_types import get_multi_plot  # type: ignore

import opencosmo as oc
from opencosmo.analysis import create_yt_dataset

if TYPE_CHECKING:
    from matplotlib.colors import Normalize
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
    yt_ds: Optional[Any] = None,
    projection_axis: Optional[str] = "z",
    north_vector: Optional[list[float]] = None,
    length_scale: Optional[str] = "top left",
    text_color: Optional[str] = "lightgray",
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

    if yt_ds is not None:
        yt_dataset_provided = True
    else:
        yt_dataset_provided = False

    if len(params["fields"]) == 4:
        # if 4 fields, make a 2x2 figure
        halo_ids = ([halo_id, halo_id], [halo_id, halo_id])

        if yt_dataset_provided:
            yt_ds = ([yt_ds, yt_ds],[yt_ds, yt_ds])

        params = {key: (value[:2], value[2:]) for key, value in params.items()}

    else:
        # otherwise, do 1xN
        halo_ids = np.shape(params["fields"])[0] * [halo_id]

        if yt_dataset_provided:
            yt_ds = np.shape(params["fields"])[0] * [yt_ds]

        params = {key: [value] for key, value in params.items()}

    return halo_projection_array(
        halo_ids,
        data,
        params=params,
        length_scale=length_scale,
        width=width,
        projection_axis=projection_axis,
        text_color=text_color,
        north_vector=north_vector,
        yt_ds=yt_ds,
    )


def halo_projection_array(
    halo_ids: int | list[int] | tuple[list[int], list[int]] | np.ndarray,
    data: oc.StructureCollection,
    yt_ds: Optional[ list[Any] | tuple[list[Any], list[Any]] ] = None,
    field: Optional[Tuple[str, str]] = ("dm", "particle_mass"),
    weight_field: Optional[Tuple[str, str]] = None,
    projection_axis: Optional[str] = "z",
    north_vector: Optional[list[float]] = None,
    cmap: Optional[str] = "gray",
    cmap_norm: Optional[Normalize] = None,  # type: ignore
    zlim: Optional[Tuple[float, float]] = None,
    params: Optional[Dict[str, Any]] = None,
    length_scale: Optional[str] = None,
    text_color: Optional[str] = "lightgray",
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
        array of projections). To leave a panel in the outputted figure blank, set the corresponding
        entry into the `halo_ids` array to `None`. If `int`, a single panel is output while preserving formatting.
    data : opencosmo.StructureCollection
        OpenCosmo StructureCollection dataset containing both halo properties and particle data
        (e.g., output of ``opencosmo.open([haloproperties, sodbighaloparticles])``).
    field : tuple of str, optional
        Field to plot for all panels. Follows yt naming conventions (e.g., ``("dm", "particle_mass")``,
        ``("gas", "temperature")``). Overridden if ``params["fields"]`` is provided.
    weight_field : tuple of str, optional
        Field to weight by during projection. Follows yt naming conventions.
        Overridden if ``params["weight_fields"]`` is provided.
    projection_axis : str, int, or 3-element sequence of floats, optional
        Data is projected along this axis (``"x"``, ``"y"``, or ``"z"``), or, alternatively,
        (0, 1, or 2). ``projection_axis`` is forwarded to the `normal` parameter of `ParticleProjectionPlot`. 
        An arbitrary projection axis may be provided as a 3-element sequence of floats.
        Overridden if ``params["projection_axes"]`` is provided
    cmap : str
        Matplotlib colormap to use for all panels. Overridden if ``params["cmaps"]`` is provided.
        See https://matplotlib.org/stable/gallery/color/colormap_reference.html for named colormaps.
    cmap_norm : Normalize
        Normalization for matplotlib colormap (e.g. for setting ``norm=matplotlib.colors.SymLogNorm()``).
        If ``None``, defaults to ``matplotlib.colors.LogNorm(vmin=zlim[0], vmax=zlim[1])``
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
            - ``"cmap_norms"``: 2D array of colormap normalization method (e.g. matplotlib.colors.LogNorm())
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

    # convert to comoving because astropy's "littleh" units aren't
    # easily translatable to yt's unit conventions
    data = data.with_units("comoving")

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
            [zlim for _ in range(np.prod(fig_shape))],
            (fig_shape[0], fig_shape[1], 2)
        )

    if isinstance(projection_axis, (str, int)):
        projection_axis_ = np.full(fig_shape, projection_axis)

    elif isinstance(projection_axis, (list, tuple, np.ndarray)):
        projection_axis_ = np.reshape(
            [projection_axis for _ in range(np.prod(fig_shape))],
            (fig_shape[0], fig_shape[1], 3),
        )

    else:
        raise RuntimeError(f"`projection_axis` has unsopported type ({type(projection_axis)}).")


    if north_vector is None:
        north_vector_ = np.full(fig_shape, None)
    else:
        north_vector_ = np.reshape(
            [north_vector for _ in range(np.prod(fig_shape))],
            (fig_shape[0], fig_shape[1], 3)
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
        "projection_axes": (projection_axis_),
        "north_vectors": (north_vector_),
        "labels": (np.full(fig_shape, None)),
        "cmaps": (np.full(fig_shape, cmap)),
        "cmap_norms": (np.full(fig_shape, None)),
        "widths": (np.full(fig_shape, width)),
    }

    # Override defaults with user-supplied params (if any)
    params = params or {}

    fields = params.get("fields", default_params["fields"])
    weight_fields = params.get("weight_fields", default_params["weight_fields"])
    projection_axes = params.get("projection_axes", default_params["projection_axes"])
    north_vectors = params.get("north_vectors", default_params["north_vectors"])
    zlims = params.get("zlims", default_params["zlims"])
    labels = params.get("labels", default_params["labels"])
    cmaps = params.get("cmaps", default_params["cmaps"])
    cmap_norms = params.get("cmap_norms", default_params["cmap_norms"])
    widths = params.get("widths", default_params["widths"])

    nrow, ncol = fig_shape

    ilen, jlen = None, None

    # define figure and axes
    fig, axes, cbars = get_multi_plot(fig_shape[1], fig_shape[0], cbar_padding=0)

    # are we plotting a single halo multiple times?
    halo_ids = np.array(halo_ids)
    halo_id_previous = np.inf

    if yt_ds is not None:
        yt_dataset_provided = True
        yt_ds = np.atleast_2d(yt_ds)
    else:
        yt_dataset_provided = False


    for i in range(nrow):
        for j in range(ncol):
            halo_id = halo_ids[i][j]
            ax = axes[i][j]

            if halo_id is None:
                ax.set_facecolor("black")
                continue

            if yt_dataset_provided:
                ds = yt_ds[i][j]

                # sodbighaloparticles holds particle data out to 2*R200
                Rh = ds.domain_width[0] / 4

            else:
                # retrieve halo particle info if new halo
                if (i == 0 and j == 0) or halo_id != halo_id_previous:
                    # retrieve properties of halo
                    if len(data) > 1:
                        data_id = data.filter(oc.col("unique_tag") == halo_id)
                    else:
                        if data["halo_properties"].data["unique_tag"] != halo_id: # type: ignore
                            raise RuntimeError(f"Halo ID {halo_id} not in dataset!")
                        data_id = data
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
                ds, 
                projection_axes[i][j],
                field,
                weight_field=weight_field,
                north_vector=north_vectors[i][j],
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

            if zlim is not None:
                zmin, zmax = zlim
            else:
                zmin, zmax = None, None

            norm = cmap_norms[i][j]

            if norm is None:
                norm = LogNorm(vmin=zmin, vmax=zmax)

            ax.imshow(
                frb[field].d,
                origin="lower",
                cmap=cmaps[i][j],
                norm=norm,
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

def fig_to_rgb(fig):
    """
    Render a Matplotlib Figure to an (H, W, 3) uint8 RGB array in memory.
    """
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())  # (H, W, 4)
    return buf[..., :3]  # drop alpha channel

def _normalize(v, eps=0):
    v = np.asarray(v, dtype=float)
    
    if eps > 0:
        # nudge exact on-axis directions off-axis
        if np.allclose(v, [1, 0, 0]):
            v = np.array([1.0, eps, 0.0])
        elif np.allclose(v, [-1, 0, 0]):
            v = np.array([-1.0, eps, 0.0])
        elif np.allclose(v, [0, 1, 0]):
            v = np.array([eps, 1.0, 0.0])
        elif np.allclose(v, [0, -1, 0]):
            v = np.array([eps, -1.0, 0.0])
        elif np.allclose(v, [0, 0, 1]):
            v = np.array([eps, 0.0, 1.0])
        elif np.allclose(v, [0, 0, -1]):
            v = np.array([eps, 0.0, -1.0])

    n = np.linalg.norm(v)

    return v / n

def _rodrigues_rotate(v, axis, angle):
    """
    Rotate vector v around 'axis' by 'angle' radians (right-hand rule).
    """
    v = np.asarray(v, dtype=float)
    k = _normalize(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    vrot = v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1 - c)
    
    return vrot

def _parse_rotations(rotations: str):
    rotations = rotations.replace(" ", "")
    parts = rotations.split("+")
    factors = []
    axes = []
    for part in parts:
        if "*" in part:
            if part.count("*") > 1:
                raise RuntimeError(f'rotation "{part}" not recognized')
            fac_str, axis = part.split("*")
            factor = float(fac_str)
        else:
            factor = 1.0
            axis = part

        if axis not in ("x", "y", "z"):
            raise RuntimeError(f'rotation axis "{axis}" not recognized')

        factors.append(factor)
        axes.append(axis)
    return factors, axes

def _get_rotation_vectors(rotations, frames, normal0=(0, 0, 1), north0=(0, 1, 0)):
    normals = [normal0]
    norths = [north0]

    factors = []
    axes = []

    axis_map = {"x": np.array([1.0, 0.0, 0.0]),
                "y": np.array([0.0, 1.0, 0.0]),
                "z": np.array([0.0, 0.0, 1.0])}

    rotation_list = rotations.replace(" ", "").split("+")
    for rotation in rotation_list:
        if "*" in rotation:
            if rotation.count("*") > 1:
                raise RuntimeError(f"rotation \"{rotation}\" not recognized")
            factor, axis = rotation.split("*")
            factor = float(factor)
        else:
            factor = 1
            axis = rotation

        factors.append(factor)
        axes.append(axis)

    # loop through rotations again, and actually apply them
    for i, rotation in enumerate(rotation_list):

        # determine number of frames for this rotation (round up)
        frames_i = int(np.ceil( frames * factors[i]/sum(factors) ))

        # angular distance traveled in theta and phi
        delta_angle_i = factors[i] * 2*np.pi / frames_i

        axis = axes[i]
         
        for _ in range(frames_i):
            normals.append(_normalize(_rodrigues_rotate(normals[-1], axis_map[axis], delta_angle_i), eps=1e-3))
            norths.append(_normalize(_rodrigues_rotate(norths[-1], axis_map[axis], delta_angle_i), eps=1e-3))

    return normals, norths



def animate_halo(halo_id, data, rotations="x", frames=30, dpi=100, normal0=(0, 0, 1), north0=(0, 1, 0)):
    
    # retrieve properties of halo and load into yt
    if len(data) > 1:
        data_id = data.filter(oc.col("unique_tag") == halo_id)
    else:
        if data["halo_properties"].data["unique_tag"] != halo_id: # type: ignore
            raise RuntimeError(f"Halo ID {halo_id} not in dataset!")
        data_id = data
    
    halo_data = next(iter(data_id.objects()))

    # load particles into yt
    ds = create_yt_dataset(halo_data)

    normals, norths = _get_rotation_vectors(rotations, 
        frames=frames, 
        normal0=normal0,
        north0=north0
    )

    fig0 = visualize_halo(
        halo_id,
        data,
        projection_axis=normals[0],
        north_vector=norths[0],
        yt_ds=ds,
    )

    frame0 = fig_to_rgb(fig0)
    plt.close(fig0)

    H, W = frame0.shape[:2]

    # ---- animation "display" figure (single persistent figure) ----
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0,0,1,1])
    ax.set_axis_off()
    ax.set_aspect("auto")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    #fig.patch.set_facecolor("black")
    im = ax.imshow(frame0, interpolation="nearest")

    def update(i):
        normal = normals[i]
        north = norths[i]

        f = visualize_halo(
            halo_id,
            data,
            projection_axis=normal,
            north_vector=north,
            yt_ds=ds,
        )
        frame = fig_to_rgb(f)
        plt.close(f)  # close each per-frame figure

        im.set_data(frame)

        return (im,)

    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

    return anim
