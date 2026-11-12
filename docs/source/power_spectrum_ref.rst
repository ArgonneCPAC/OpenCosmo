Power Spectrum
==============

Opening a power spectrum file with :py:func:`opencosmo.open` returns a
:py:class:`opencosmo.PowerSpectrum`. Like a dataset, a power spectrum is
transformed by methods that return a new power spectrum —
:py:meth:`select <opencosmo.PowerSpectrum.select>`,
:py:meth:`drop <opencosmo.PowerSpectrum.drop>`,
:py:meth:`with_steps <opencosmo.PowerSpectrum.with_steps>`,
:py:meth:`with_redshifts <opencosmo.PowerSpectrum.with_redshifts>`, and
:py:meth:`with_units <opencosmo.PowerSpectrum.with_units>` — and the spectra are
read with :py:meth:`get_data <opencosmo.PowerSpectrum.get_data>`. See
:doc:`power_spectra` for a guide.

.. autoclass:: opencosmo.PowerSpectrum
   :members:
   :exclude-members: close
   :member-order: bysource
