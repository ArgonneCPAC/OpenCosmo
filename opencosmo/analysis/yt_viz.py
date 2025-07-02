import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import yt # type: ignore
from unyt import unyt_quantity # type: ignore
from yt.visualization.base_plot_types import get_multi_plot # type: ignore
from yt.visualization.plot_container import (  # type: ignore
    ImagePlotContainer,
    PlotContainer,
)
from yt.visualization.plot_window import PlotWindow, NormalPlot  # type: ignore

import opencosmo as oc
from opencosmo.analysis import create_yt_dataset 



def ParticleProjectionPlot(*args, **kwargs) -> NormalPlot:
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
    return yt.ParticleProjectionPlot(*args, **kwargs)


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


def SlicePlot(*args, **kwargs) -> PlotWindow:
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


def ProfilePlot(*args, **kwargs) -> PlotWindow:
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


def PhasePlot(*args, **kwargs) -> PlotWindow:
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



def visualize_halo(halo_id, data, length_scale="top left", width=4.0):
    """
    Creates a 2x2 figure showing particle projections of dark matter, stars, gas, and gas temperature
    for given halo. To customize the arrangement of panels, fields, colormaps, etc., see 
    :func:`visualize_halos_custom`. Each panel is an 800x800 pixel array.
 

    Parameters
    ----------
    halo_id : int
        Identifier of the halo to be visualized.
    data : opencosmo.Dataset
        OpenCosmo dataset containing both halo properties and particle data
        ( e.g. output of opencosmo.open_linked_files([haloproperties, sodbighaloparticles]) )
    length_scale : str or None
        Optionally add a horizontal bar denoting length scale in Mpc
        Options:
        - `"top left"` -- add to top left panel
        - `"top right"` -- add to top right panel
        - `"bottom left"` -- add to bottom left panel
        - `"bottom right"` -- add to bottom right panel
        - `"all"` -- add to all panels
        - None -- no length scale on all panels
    width : float, optional
        Width of each projection panel in units of R200 for the halo. Default is 4.0.

    """

    halo_ids = ( [halo_id, halo_id], [halo_id, halo_id] )
    params = {
        "fields": (
            [("dm", "particle_mass"), ("star", "particle_mass")],
            [("gas", "particle_mass"), ("gas", "temperature")]
        ),
        "weight_fields": (
            [None, None],
            [None, ("gas", "density")]
        ),
        "zlims": (
            [None, None],
            [None, (1e7, 1e8)]
        ),
        "labels": (
            ["Dark Matter", "Stars"],
            ["Gas", "Gas Temperature"]
        ),
        "cmaps": (
            ["gray", "bone"],
            ["viridis", "inferno"]
        ),
    }

    return multipanel_halo_projections(halo_ids, data, params=params, 
        length_scale=length_scale, width=width)



def multipanel_halo_projections(halo_ids, data, field=("dm", "particle_mass"), 
                    weight_field=None, cmap="gray", zlim=None, params=None, 
                    length_scale=None, smooth_gas_fields=False, width=4.0):
    """
    Creates a multipanel figure of projections for different fields.
    By default, creates an arrangement of dark matter particle projections with the 
    same shape as `halo_ids`. Each panel is an 800x800 pixel array.
    
    Customizable -- can change which fields are plotted for which halos, their order, weighting,
        etc. with `params`

    Parameters
    ----------
    halo_ids : 2D array
        Identifier of the halo(s) to be visualized. Shape of this array sets the shape
        of the figure (e.g. if halo_ids is a 2x3 array, the outputted figure will be a 2x3 array
        of projections)
    data : opencosmo StructuredCollection
        OpenCosmo dataset containing both halo properties and particle data
        ( e.g. output of opencosmo.open_linked_files([haloproperties, sodbighaloparticles]) )
    field : plot this field for all panels. Follows yt naming conventions. 
            Overwritten if `params["fields"]` is set 
    weight_field : weight by this field during projection for all panels. Follows yt naming conventions. 
            Overwritten if `params["weight_fields"]` is set 
    cmap : matplotlib colormap to use for all panels. Overwritten if `params["cmaps"]` is set.
           See https://matplotlib.org/stable/gallery/color/colormap_reference.html
    zlim : tuple(int)
        colorbar limits for `field`. Overwritten if `params["cmaps"]` is set
    length_scale : str or None
        Optionally add a horizontal bar denoting length scale in Mpc
        Options:
        - `"top left"` -- add to top left panel
        - `"top right"` -- add to top right panel
        - `"bottom left"` -- add to bottom left panel
        - `"bottom right"` -- add to bottom right panel
        - `"all"` -- add to all panels
        - None -- no length scale on all panels

    params : dict, optional
        Dictionary of customization parameters for the projection panels. Overrides
        defaults. All inputs must be 2D arrays with the same shape.

        Keys may include:
            - 'fields': 2D array of fields to plot, following yt naming conventions
            - 'weight_fields': 2D array of weight fields for projection. No weighting if None
            - 'zlims': 2D array of colorbar limits. Colorbars are log-scaled
            - 'labels': 2D array of panel labels. No label if None
            - 'cmaps': 2D array of matplotlib colormaps to use for panels. 
                       See https://matplotlib.org/stable/gallery/color/colormap_reference.html
    width : float, optional
        Width of each projection panel in units of R200 for the halo. Default is 4.0.
    """

    # determine shape of figure
    fig_shape = np.shape(halo_ids)

    # Default plotting parameters
    if weight_field is None:
        weight_field_ = np.full(fig_shape, None)
    else:
        weight_field_ = np.reshape( 
            [weight_field for _ in range(np.prod(fig_shape))], 
            (fig_shape[0], fig_shape[1], 2) 
        )

    if zlim is None:
        zlim_ = np.full(fig_shape, None)
    else:
        zlim_ = np.reshape(
            [zlim for _ in range(np.prod(fig_shape))],
            (fig_shape[0], fig_shape[1], 2)
        )

    default_params = {
        "fields": (
            np.reshape( [field for _ in range(np.prod(fig_shape))], 
                (fig_shape[0], fig_shape[1], 2) ) 
        ),
        "weight_fields": (
            weight_field_ 
        ),
        "zlims": (
            zlim_
        ),
        "labels": (
            np.full(fig_shape, None)
        ),
        "cmaps": (
            np.full(fig_shape, cmap)
        ),
    }

    # Override defaults with user-supplied params (if any)
    params = params or {}

    fields = params.get("fields", default_params["fields"])
    weight_fields = params.get("weight_fields", default_params["weight_fields"])
    zlims = params.get("zlims", default_params["zlims"])
    labels= params.get("labels", default_params["labels"])
    cmaps = params.get("cmaps", default_params["cmaps"])

    nrow, ncol = fig_shape

    ilen, jlen = None, None
    match length_scale:
        case "top left":
            ilen, jlen = 0, 0
        case "top right":
            ilen, jlen = 0, ncol-1
        case "bottom left":
            ilen, jlen = nrow-1, 0
        case "bottom right":
            ilen, jlen = nrow-1, ncol-1
    
    

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
                halo_data = next(data_id.objects())
                
                # load particles into yt
                ds = create_yt_dataset(halo_data)

            halo_properties = halo_data['halo_properties']

            Rh = unyt_quantity.from_astropy(halo_properties['sod_halo_radius'])

            field, weight_field, zlim = tuple(fields[i][j]), weight_fields[i][j], zlims[i][j]

            if weight_field is not None:
                weight_field = tuple(weight_field)
            if zlim is not None:
                zlim = tuple(zlim)

            label = labels[i][j]

            if smooth_gas_fields and field[0] == "gas":
                proj = ProjectionPlot(ds,'z',field, weight_field = weight_field)
            else:
                proj = ParticleProjectionPlot(ds,'z',field, weight_field = weight_field)

            proj.set_background_color(field, color='black')
            proj.set_width(width*Rh)

            # fetch figure buffer (2D array of pixel values) 
            # and re-plot on each panel with imshow
            frb = proj.frb
            
            ax = axes[i][j]

            if zlim is not None:
                zmin, zmax = zlim
            else:
                zmin, zmax = None, None

            ax.imshow(frb[field], origin="lower", cmap=cmaps[i][j], 
                norm=LogNorm(vmin=zmin, vmax=zmax))
            ax.set_facecolor("black")

            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            cbars[i].set_visible(False)

            if label is not None:
                # add panel label
                ax.text(
                    0.06, 0.94,
                    label,
                    transform=ax.transAxes,
                    ha='left', va='top',
                    fontsize=12,
                    fontfamily='DejaVu Serif',
                    color = "grey"
                )

            if length_scale is not None:
                if (i==ilen and j==jlen) or length_scale=="all":
                    # add length scale, assuming
                    # panel is 800 pixels wide
                    scalebar = AnchoredSizeBar(
                        ax.transData,
                        800/(width*Rh.d),            
                        '1 Mpc',               
                        'lower right',         
                        pad=0.4, label_top=False,
                        sep=10,
                        color='grey',
                        frameon=False,
                        size_vertical=1, 
                    )
                    ax.add_artist(scalebar)

            halo_id_previous = halo_id

    return fig
