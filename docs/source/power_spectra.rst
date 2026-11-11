Power Spectra
=============

In the context of cosmology, a power spectrum describes the magnitude of density fluctuations for a given component as a function of size. Specifically,
it is a `fourier transform of the correlation function of a given density field <https://en.wikipedia.org/wiki/Matter_power_spectrum>`_.

A power spectrum file holds these spectra for one simulation: the spectrum of each component at
each of a list of steps, on one set of wavenumber bins. Opening such a file returns
a :py:class:`PowerSpectrum <opencosmo.PowerSpectrum>`.

.. code-block:: python

   import opencosmo as oc

   ps = oc.open("hydro_power_spectrum.hdf5")
   print(ps.components)   # ('agn', 'bar', 'cdm', 'gas', 'sfgas', 'star', 'total', 'wind')
   print(ps.steps)        # the simulation steps
   print(ps.redshifts)    # the redshift of each step

   k = ps.k                                        # shape (n_k,)
   pk_cdm = ps.select("cdm").with_steps(624).get_data()      # shape (1, n_k)
   pk_cdm = ps.select("cdm").with_redshifts(0.5).get_data()  # the step closest to z = 0.5, within 0.01
   pk_all = ps.select("total").get_data()["total"]                  # shape (n_steps, n_k)

Like a dataset, a power spectrum is transformed by methods that return a new
power spectrum: :py:meth:`select <opencosmo.PowerSpectrum.select>` and
:py:meth:`drop <opencosmo.PowerSpectrum.drop>` choose components, and
:py:meth:`with_steps <opencosmo.PowerSpectrum.with_steps>` and
:py:meth:`with_redshifts <opencosmo.PowerSpectrum.with_redshifts>` choose steps.
``get_data`` then returns every remaining component and step, keyed by component
name. By default it is returned as an astropy QTable, but this can be changed with
the ``format`` keyword, just like
:py:meth:`Dataset.get_data <opencosmo.Dataset.get_data>`:

.. code-block:: python

   pks = ps.get_data()                                # QTable with one column per component
   pks = ps.select("cdm", "gas").get_data()           # same, for just those components
   pks = ps.drop("agn", "wind").get_data()            # same, without those components
   pks = ps.get_data(format="numpy")                  # dictionary of numpy arrays

When more than one step is selected, the output covers every selected step and
includes a "redshift" column. A single component at a single step is returned on
its own, like a single-column :py:meth:`Dataset.get_data <opencosmo.Dataset.get_data>`
call:

.. code-block:: python

   pks = ps.with_redshifts([0, 0.5, 1]).get_data()  # rows are the closest steps
   pks["redshift"]                                  # their redshifts

Gravity-only simulations have only the "total" component. Hydrodynamic simulations
also have the spectra of the dark matter, the baryons, and several subsets of the
baryons. Each component is normalized to its own mean density. The spectrum of a
component is NaN at the steps where the component has no particles, for example
the black holes at early steps.

``n_modes`` is the number of independent modes in each bin, so the sampling error
of the power in a bin is ``P / sqrt(n_modes)``:

.. code-block:: python

   import numpy as np

   error = pk_cdm / np.sqrt(ps.n_modes)

:py:attr:`ps.parameters <opencosmo.PowerSpectrum.parameters>` returns a
dictionary of the parameters used during measurement, such as the grid size, the
mass assignment and the bins. Check ``ps.parameters["shot_noise_subtracted"]`` to
see whether the shot noise has been subtracted from the spectra. As for a dataset,
the cosmology and the simulation parameters are available as ``ps.cosmology`` and
``ps.simulation``.

Units
-----

A power spectrum opens in the "comoving" unit convention, like a dataset, with
wavenumbers in 1/Mpc and spectra in Mpc^3. :py:meth:`with_units
<opencosmo.PowerSpectrum.with_units>` allows you to change to the "scalefree" convention, with
wavenumbers in h/Mpc and spectra in (Mpc/h)^3. Power spectra are
defined in comoving coordinates, so attempting to change to the "physical" convention will
raise an error.

Selecting a redshift
--------------------

``with_redshifts(z)`` selects the step whose redshift is closest to ``z``. You may
also pass a list of redshifts to select several steps. Simulation steps are often
not at exactly round redshifts, so ``with_redshifts(0.5)`` may select a step at
z = 0.4987, for example. If no step is within 0.01 of ``z``, the call raises a
``ValueError`` that names the closest steps. Pass ``tolerance`` to change this
limit, for example ``with_redshifts(0.5, tolerance=0.05)``. Use ``with_steps`` to
select a step by number.
