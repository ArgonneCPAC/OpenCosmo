Power Spectra
=============

HACC measures the matter power spectrum while a simulation runs. A power spectrum
file holds these spectra for one simulation: the spectrum of each component at
each of a list of steps, on one set of wavenumber bins. Opening such a file returns
a :py:class:`PowerSpectrum <opencosmo.PowerSpectrum>`.

.. code-block:: python

   import opencosmo as oc

   ps = oc.open("power_spectrum.hdf5")
   print(ps.components)   # ('agn', 'bar', 'cdm', 'gas', 'sfgas', 'star', 'total', 'wind')
   print(ps.steps)        # the simulation steps
   print(ps.redshifts)    # the redshift of each step

   k = ps.k                                  # shape (n_k,)
   pk_cdm = ps.get_data("cdm", step=624)     # shape (n_k,)
   pk_cdm = ps.get_data("cdm", redshift=0.5) # the step closest to z = 0.5
   pk_all = ps.get_data("total")             # shape (n_steps, n_k)

A gravity-only simulation has only the "total" component. A hydrodynamic simulation
also has the spectra of the dark matter, the baryons and several subsets of the
baryons. Each component is normalized to its own mean density. The spectrum of a
component is NaN at the steps where the component has no particles, for example
the black holes at early steps.

``n_modes`` is the number of independent modes in each bin, so the sampling error
of the power in a bin is ``P / sqrt(n_modes)``:

.. code-block:: python

   import numpy as np

   error = pk_cdm / np.sqrt(ps.n_modes)

HACC does not subtract the shot noise. The header block ``power_spectrum`` records
how HACC measured the spectra, such as the grid size, the mass assignment and the
bins. As for a dataset, the cosmology and the simulation parameters are available
as ``ps.cosmology`` and ``ps.simulation``.

Units
-----

A power spectrum opens in the "comoving" unit convention, like a dataset, with
wavenumbers in 1/Mpc and spectra in Mpc^3. :py:meth:`with_units
<opencosmo.PowerSpectrum.with_units>` changes the convention. In the "physical"
convention, the wavenumbers and the spectrum of each step are converted with the
scale factor of that step. The physical wavenumbers therefore differ between
steps, and :py:attr:`k <opencosmo.PowerSpectrum.k>` has the shape (n_steps, n_k):

.. code-block:: python

   physical = ps.with_units("physical")
   i = 0
   k = physical.k[i]
   pk = physical.get_data("total", step=physical.steps[i])
