from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from opencosmo.power_spectrum import state as psst

if TYPE_CHECKING:
    import astropy.units as u
    import numpy as np

    from opencosmo.header import OpenCosmoHeader
    from opencosmo.power_spectrum.state import PowerSpectrumState


class PowerSpectrum:
    """
    The matter power spectra that HACC measured while a simulation ran. A power
    spectrum file holds the spectrum of each component at each of a list of steps,
    on one set of wavenumber bins.

    Like a :py:class:`Dataset <opencosmo.Dataset>`, a power spectrum opens in the
    "comoving" unit convention (there is no "physical" convention for spectra), and its header parameters such as ``cosmology`` and
    ``simulation`` are available as attributes. HACC's measurement settings are in
    ``power_spectrum``.

    .. code-block:: python

        import opencosmo as oc

        ps = oc.open("power_spectrum.hdf5")
        k = ps.k
        pk_cdm = ps.get_data("cdm", redshift=0)
    """

    def __init__(self, state: PowerSpectrumState):
        self.__state = state

    def __repr__(self) -> str:
        steps = self.steps
        return (
            f"OpenCosmo PowerSpectrum ({len(steps)} steps from {steps[0]} to "
            f"{steps[-1]}, {len(self.n_modes)} wavenumber bins, components: "
            f"{', '.join(self.components)})"
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        return None

    def close(self):
        return None

    def __getattr__(self, key: str):
        try:
            return self.header.parameters[key]
        except KeyError:
            return object.__getattribute__(self, key)

    def __dir__(self):
        return [*self.header.parameters, *super().__dir__()]

    @property
    def header(self) -> OpenCosmoHeader:
        """
        The header of the file.

        Returns
        -------
        header: opencosmo.header.OpenCosmoHeader
        """
        return self.__state.header

    @property
    def components(self) -> tuple[str, ...]:
        """
        The components that have a spectrum in this file, for example "total" for
        all particles and "cdm" for the dark matter particles. A gravity-only
        simulation has only "total".

        Returns
        -------
        components: tuple[str, ...]
        """
        return self.__state.components

    @property
    def steps(self) -> np.ndarray:
        """
        The simulation steps of the spectra, in increasing order.

        Returns
        -------
        steps: np.ndarray
        """
        return self.__state.steps.copy()

    @property
    def redshifts(self) -> np.ndarray:
        """
        The redshift of each step.

        Returns
        -------
        redshifts: np.ndarray
        """
        return self.__state.redshifts.copy()

    @property
    def k(self) -> u.Quantity | np.ndarray:
        """
        The mean wavenumber of the modes in each bin, with shape (n_k,). The
        wavenumbers are comoving and the same at every step.

        Returns
        -------
        k: astropy.units.Quantity
        """
        return psst.get_k(self.__state)

    @property
    def n_modes(self) -> np.ndarray:
        """
        The number of independent modes in each bin. The sampling error of the
        power in a bin is ``P / sqrt(n_modes)``.

        Returns
        -------
        n_modes: np.ndarray
        """
        return self.__state.n_modes.copy()

    def get_data(
        self,
        component: str = "total",
        step: int | None = None,
        redshift: float | None = None,
    ) -> u.Quantity | np.ndarray:
        """
        Get the power spectrum of one component. Without a step or redshift, this
        returns the spectra of all steps as an array of shape (n_steps, n_k). With
        a step or a redshift, it returns the spectrum at that step, with shape
        (n_k,).

        The spectrum of a component is NaN at the steps where it has no
        particles, for example the black holes at early steps.

        Parameters
        ----------
        component: str, default="total"
            One of :py:attr:`components`.
        step: int, optional
            The step of the spectrum. Must be one of :py:attr:`steps`.
        redshift: float, optional
            The redshift of the spectrum. This selects the step whose redshift in
            :py:attr:`redshifts` is closest to the given value, and it fails if
            that step is more than 0.01 away. HACC steps lie within a few
            thousandths of round redshifts, so ``redshift=0.5`` selects the step at
            z = 0.4987 in the SCIDAC simulations. Cannot be combined with ``step``.

        Returns
        -------
        pk: astropy.units.Quantity

        Raises
        ------
        ValueError
            If the component or step is not in the file, if no step is within
            0.01 of the redshift (the message names the closest steps), or if both
            a step and a redshift are given.
        """
        index = psst.step_index(self.__state, step, redshift)
        return psst.get_spectrum(self.__state, component, index)

    def with_units(
        self,
        convention: Literal["scalefree", "comoving", "unitless"],
    ) -> PowerSpectrum:
        """
        Create a new power spectrum with a different unit convention. See
        :doc:`units`. Power spectra are defined in comoving coordinates, so the
        "physical" convention is not available. Use "comoving" for values without
        factors of h.

        Parameters
        ----------
        convention : str
            One of "comoving", "scalefree", or "unitless".

        Returns
        -------
        power_spectrum : PowerSpectrum

        Raises
        ------
        ValueError
            If the convention is "physical" or unknown.
        """
        return PowerSpectrum(psst.with_units(self.__state, convention))
