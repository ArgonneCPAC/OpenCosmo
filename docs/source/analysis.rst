Analyzing Particle Data with yt
===============================

`yt <https://github.com/yt-project/yt>`_ is an open-source Python package for analyzing and visualizing volumetric simulation data. Although yt was originally designed with AMR (Adaptive Mesh Refinement) codes in mind, support for SPH (Smoothed Particle Hydrodynamics) data is continually improving. As of yt version 4.4, most core functionality works reliably with SPH data, though some features may still require workarounds. In many cases, this involves depositing particle data onto a mesh using a ``YTArbitraryGrid`` object before passing it to specific yt functions.

In OpenCosmo, you can load particle datasets into yt using :func:`opencosmo.analysis.create_yt_dataset`. 
This is effectly doing the same as `yt.load`, however, we have opted to use the OpenCosmo toolkit
to handle the initial data selection.

Here is an example for how to use `create_yt_dataset` to load a selection of data into yt and make a simple projection

.. code-block:: python

    from opencosmo.analysis import create_yt_dataset, ParticleProjectionPlot

    # select a random halo
    with oc.open("haloparticles.hdf5").take(1, at="random") as data:
        for halo_particles in data.objects():
            # get yt data container
            ds = create_yt_dataset(halo_particles)

            # list all fields
            print(ds.derived_field_list)

            # project DM particle mass
            ParticleProjectionPlot(ds, 'z', ('dm', 'particle_mass')).save()
            

For convenience, OpenCosmo includes wrappers for several commonly used yt plotting functions, including:

- :func:`opencosmo.analysis.ParticleProjectionPlot` (wraps ``yt.ParticleProjectionPlot``)
- :func:`opencosmo.analysis.ProjectionPlot` (wraps ``yt.ProjectionPlot``)
- :func:`opencosmo.analysis.SlicePlot` (wraps ``yt.SlicePlot``)
- :func:`opencosmo.analysis.ProfilePlot` (wraps ``yt.ProfilePlot``)
- :func:`opencosmo.analysis.PhasePlot` (wraps ``yt.PhasePlot``)

These wrappers follow the same naming conventions as the original yt functions and have been verified to work out-of-the-box with HACC SPH data.

For an overview of yt’s broader functionality, refer to the official `yt documentation <https://yt-project.org/doc/index.html>`_.

For introductory tutorials, see:

- `Making Simple Plots <https://yt-project.org/doc/cookbook/simple_plots.html>`_
- `A Few Complex Plots <https://yt-project.org/doc/cookbook/complex_plots.html>`_


Simulating X-ray Emission with pyXSIM
=====================================

To include synthetic X-ray emissivity and luminosity fields in your yt dataset, you can enable the ``compute_xray_fields`` flag when calling :func:`opencosmo.analysis.create_yt_dataset`. This integrates with `pyXSIM <https://hea-www.cfa.harvard.edu/~jzuhone/pyxsim/>`_, a toolkit for generating synthetic X-ray observations from simulation data.

When ``compute_xray_fields=True``, the function internally creates a :class:`pyxsim.CIESourceModel` using the particle data and attaches the following derived fields to the `yt` dataset:

- X-ray emissivity per particle
- X-ray luminosity in a user-specified energy band
- Any additional fields required for photon sampling (e.g., emission measure)

You can also pass model-specific configurations via the ``source_model_kwargs`` argument, which is forwarded directly to the :class:`pyxsim.CIESourceModel` constructor. Common options include:

- ``emin`` (float): Minimum photon energy in keV (default: 0.1)
- ``emax`` (float): Maximum photon energy in keV (default: 10.0)
- ``nbins`` (int): Number of bins across the energy band (default: 1000)
- ``model`` (str): which emission model to use (default: "apec")

For the full list of options, see `CIESourceModel <https://hea-www.cfa.harvard.edu/~jzuhone/pyxsim/api/source_models.html#pyxsim.source_models.thermal_sources.CIESourceModel>`_.

If ``return_source_model=True``, the function will return a 2-tuple ``(ds, source_model)``, where ``source_model`` is the ``CIESourceModel`` instance. This allows further customization or photon generation using pyXSIM directly.

We will now edit the code-block from before to compute X-ray luminosities:

.. code-block:: python

    from opencosmo.analysis import create_yt_dataset, ParticleProjectionPlot

    # set source model parameters
    source_model_kwargs = {
        "emin": 0.1, # keV
        "emax": 10.0 # keV
    }

    # select a random halo
    with oc.open("haloparticles.hdf5").take(1, at="random") as data:
        for halo_particles in data.objects():
            # get yt data container
            ds, source_model = create_yt_dataset(halo_particles, 
                compute_xray_fields = True, return_source_model = True)

            # list all fields
            print(ds.derived_field_list)

            # project X-ray luminosity in the specified band
            ParticleProjectionPlot(ds, 'z', ('gas', 'xray_luminosity_0.1_10.0_keV')).save()

