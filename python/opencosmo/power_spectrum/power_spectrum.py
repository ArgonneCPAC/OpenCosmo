from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Literal

import numpy as np

from opencosmo.units import UnitConvention
from opencosmo.units.handler import make_unit_handler_from_unit_strings

if TYPE_CHECKING:
    import astropy.units as u

    from opencosmo.header import OpenCosmoHeader
    from opencosmo.io.iopen import DatasetTarget
    from opencosmo.io.schema import Schema
    from opencosmo.units.handler import UnitHandler

PREFIX = "pk_"
# The largest difference between a requested redshift and the step it selects.
# HACC steps sit within a few thousandths of the round redshifts users ask for.
REDSHIFT_TOLERANCE = 0.01


@dataclass(frozen=True, eq=False, repr=False)
class PowerSpectrum:
    """
    The matter power spectra that HACC measured while a simulation ran. A power
    spectrum file holds the spectrum of each component at each of a list of steps,
    on one set of wavenumber bins.

    Like a :py:class:`Dataset <opencosmo.Dataset>`, a power spectrum opens in the
    "comoving" unit convention (there is no "physical" convention for spectra), and
    its header parameters such as ``cosmology`` and ``simulation`` are available as
    attributes. HACC's measurement settings are in ``power_spectrum``.

    .. code-block:: python

        import opencosmo as oc

        ps = oc.open("power_spectrum.hdf5")
        k = ps.k
        pk_cdm = ps.get_data("cdm", redshift=0)
    """

    # Power spectra are small, so the whole file is read when it is opened. Unlike
    # datasets and collections, there is no lazy backend state behind this class.
    header: OpenCosmoHeader
    """The header of the file."""
    components: tuple[str, ...]
    """The components that have a spectrum in this file, for example "total" for
    all particles and "cdm" for the dark matter particles. A gravity-only
    simulation has only "total"."""
    steps: np.ndarray
    """The simulation steps of the spectra, in increasing order. Read-only."""
    redshifts: np.ndarray
    """The redshift of each step. Read-only."""
    n_modes: np.ndarray
    """The number of independent modes in each bin. The sampling error of the
    power in a bin is ``P / sqrt(n_modes)``. Read-only."""
    _k: np.ndarray
    # Shape (n_components, n_steps, n_k), in the order of ``components``.
    _spectra: np.ndarray
    _unit_handler: UnitHandler

    def __repr__(self) -> str:
        return (
            f"OpenCosmo PowerSpectrum ({len(self.steps)} steps from {self.steps[0]} "
            f"to {self.steps[-1]}, {len(self.n_modes)} wavenumber bins, components: "
            f"{', '.join(self.components)})"
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        return None

    def close(self):
        return None

    # The object and its arrays are immutable, so a copy can share them. A real
    # deep copy would also turn the read-only arrays writeable.
    def __copy__(self) -> PowerSpectrum:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> PowerSpectrum:
        return self

    def __getattr__(self, key: str):
        # Fields are missing while the object is being unpickled or copied. Refusing
        # them avoids recursing through ``header``.
        if key in _FIELDS:
            raise AttributeError(key)
        try:
            return self.header.parameters[key]
        except KeyError:
            return object.__getattribute__(self, key)

    def __dir__(self):
        return [*self.header.parameters, *object.__dir__(self)]

    def make_schema(self, path: str) -> Schema:
        """
        Prep to write the power spectrum. This should not be called directly by
        the user. The opencosmo.write file writer automatically handles the file
        context.

        Parameters
        ----------
        path : str
            The path of the file the power spectrum will be written to.

        Returns
        -------
        schema : opencosmo.io.schema.Schema
        """
        from opencosmo.io.schema import FileEntry, make_schema
        from opencosmo.io.writer import ColumnCombineStrategy, ColumnWriter

        # Columns are written with their base-convention units, as stored on disk.
        units = self._unit_handler.base_units

        def writer(name: str, values: np.ndarray) -> ColumnWriter:
            unit = units.get(name)
            attrs = {} if unit is None else {"unit": str(unit)}
            # Every rank holds the identical spectrum, so under MPI only the
            # first rank writes each column.
            return ColumnWriter.from_numpy_array(
                values, strategy=ColumnCombineStrategy.EXACT, attrs=attrs
            )

        columns = {
            "step": writer("step", self.steps),
            "redshift": writer("redshift", self.redshifts),
            "k": writer("k", self._k),
            "n_modes": writer("n_modes", self.n_modes),
        }
        for i, name in enumerate(self.components):
            columns[f"{PREFIX}{name}"] = writer(f"{PREFIX}{name}", self._spectra[i])
        spectrum_schema = make_schema(
            "power_spectrum", FileEntry.COLUMNS, columns=columns
        )
        return make_schema(
            path.split("/")[-1],
            FileEntry.POWER_SPECTRUM,
            children={
                "power_spectrum": spectrum_schema,
                "header": self.header.dump(),
            },
        )

    @property
    def k(self) -> u.Quantity | np.ndarray:
        """
        The mean wavenumber of the modes in each bin, with shape (n_k,). The
        wavenumbers are comoving and the same at every step.

        Returns
        -------
        k: astropy.units.Quantity
        """
        return _apply_units(self._unit_handler, "k", self._k)

    def get_data(
        self,
        components: str | list[str] = "all",
        step: int | None = None,
        redshift: float | list[float] | None = None,
        format: str = "astropy",
    ):
        """
        Get the power spectrum of one or more components. Without a step or
        redshift, each spectrum covers all steps as an array of shape
        (n_steps, n_k). With a single step or redshift, it covers that step only,
        with shape (n_k,). With a list of redshifts, it covers those steps as an
        array of shape (n_selected, n_k).

        When more than one step is selected, the output also has a "redshift"
        column with the redshift of each step, with shape (n_steps,). The spectrum
        of a component is NaN at the steps where it has no
        particles, for example the black holes at early steps.

        Parameters
        ----------
        components: str or list of str, default="all"
            The components to get: "all", one of :py:attr:`components`, or a list
            of them. "all" and a list return the spectra of every requested
            component keyed by name in the output format's container; a single
            named component is returned on its own, like a single-column
            :py:meth:`Dataset.get_data <opencosmo.Dataset.get_data>` call.
        step: int, optional
            The step of the spectrum. Must be one of :py:attr:`steps`.
        redshift: float or list of float, optional
            The redshift(s) of the spectrum. Each value selects the step whose
            redshift in :py:attr:`redshifts` is closest to it, and it fails if
            that step is more than 0.01 away. HACC steps lie within a few
            thousandths of round redshifts, so ``redshift=0.5`` selects the step at
            z = 0.4987 in the SCIDAC simulations. Cannot be combined with ``step``.
        format: str, default="astropy"
            The format to output the data in. Currently supported are "astropy",
            "numpy", "pandas", "polars", "arrow", and "jax". The optional formats
            require the relevant library to be installed.

        Returns
        -------
        pk: astropy.units.Quantity, or a container of them
            A single component at a single step is returned on its own; otherwise
            the result is the output format's container (QTable, DataFrame, dict,
            ...) keyed by component name, with a "redshift" column when several
            steps are selected.

        Raises
        ------
        ValueError
            If a component or step is not in the file, if no step is within
            0.01 of a redshift (the message names the closest steps), if both
            a step and a redshift are given, or if the format is unknown.
        ImportError
            If the format requires a package that is not installed.
        """
        from opencosmo.dataset.formats import convert_data, verify_format

        names = _resolve_components(self.components, components)
        selection = _step_selection(self.steps, self.redshifts, step, redshift)
        data = {}
        for name in names:
            spectrum = self._spectra[self.components.index(name)]
            # The stored arrays are read-only, so slices and views of them can be
            # handed out without copying.
            values = spectrum if selection is None else spectrum[selection]
            data[f"{PREFIX}{name}"] = values
        data = self._unit_handler.apply_units(data, unit_kwargs={})
        data = {name.removeprefix(PREFIX): value for name, value in data.items()}
        if selection is None or isinstance(selection, list):
            data["redshift"] = (
                self.redshifts if selection is None else self.redshifts[selection]
            )
        verify_format(format)
        return convert_data(data, format, wrap_single=len(data) != 1)

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
        unit_convention = UnitConvention(convention)
        if unit_convention == UnitConvention.PHYSICAL:
            raise ValueError(
                "Power spectra are defined in comoving coordinates and have no "
                "physical units. Use the 'comoving' convention for values without "
                "factors of h."
            )
        return replace(
            self,
            header=self.header.with_units(unit_convention),
            _unit_handler=self._unit_handler.with_convention(unit_convention),
        )


_FIELDS = frozenset(field.name for field in fields(PowerSpectrum))


def read_power_spectrum(target: DatasetTarget) -> PowerSpectrum:
    """Read the /power_spectrum group of a discovered power spectrum file."""
    group = target.group["power_spectrum"]
    names = [name for name in target.column_names if name.startswith(PREFIX)]
    unit_handler = make_unit_handler_from_unit_strings(
        target.column_units, target.header, UnitConvention.COMOVING
    )
    return PowerSpectrum(
        header=target.header.with_units(UnitConvention.COMOVING),
        steps=__read_only(group["step"][:]),
        redshifts=__read_only(group["redshift"][:]),
        _k=__read_only(group["k"][:]),
        n_modes=__read_only(group["n_modes"][:]),
        components=tuple(name.removeprefix(PREFIX) for name in names),
        _spectra=__read_only(np.stack([group[name][:] for name in names])),
        _unit_handler=unit_handler,
    )


def __read_only(array: np.ndarray) -> np.ndarray:
    # A view of a read-only array cannot be made writeable again, unlike the array
    # that owns the data.
    array.flags.writeable = False
    return array.view()


# Single underscore: these are called from the class body, where a double-underscore
# name would be mangled.
def _apply_units(
    unit_handler: UnitHandler, name: str, values: np.ndarray
) -> np.ndarray | u.Quantity:
    return unit_handler.apply_units({name: values}, unit_kwargs={})[name]


def _resolve_components(
    available: tuple[str, ...], components: str | list[str]
) -> list[str]:
    if isinstance(components, str):
        requested = list(available) if components == "all" else [components]
    else:
        requested = list(components)
        if not requested:
            raise ValueError(
                "Pass at least one component. Available components: "
                f"{', '.join(available)}"
            )
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(
            f"This power spectrum has no component {unknown[0]!r}. Available "
            f"components: {', '.join(available)}"
        )
    return list(dict.fromkeys(requested))


def _step_selection(
    steps: np.ndarray,
    redshifts: np.ndarray,
    step: int | None,
    redshift: float | list[float] | None,
) -> int | list[int] | None:
    """None for all steps, an index for one step, or a list of indices."""
    if step is not None and redshift is not None:
        raise ValueError("Pass either a step or a redshift, not both")
    if step is not None:
        matches = np.flatnonzero(steps == step)
        if len(matches) == 0:
            raise ValueError(
                f"Step {step} is not in this power spectrum. Available steps: "
                f"{', '.join(map(str, steps))}"
            )
        return int(matches[0])
    if redshift is not None:
        if isinstance(redshift, (list, tuple, np.ndarray)):
            return [_redshift_index(steps, redshifts, z) for z in redshift]
        return _redshift_index(steps, redshifts, redshift)
    return None


def _redshift_index(steps: np.ndarray, redshifts: np.ndarray, redshift: float) -> int:
    """The index of the step closest to ``redshift``, within REDSHIFT_TOLERANCE."""
    if not np.isfinite(redshift):
        raise ValueError(f"The redshift must be finite, got {redshift}")
    distance = np.abs(redshifts - redshift)
    index = int(np.argmin(distance))
    if distance[index] <= REDSHIFT_TOLERANCE:
        return index
    below = np.flatnonzero(redshifts < redshift)
    above = np.flatnonzero(redshifts > redshift)
    closest = [
        i
        for i in (
            below[np.argmax(redshifts[below])] if len(below) else None,
            above[np.argmin(redshifts[above])] if len(above) else None,
        )
        if i is not None
    ]
    candidates = " and ".join(
        f"step {steps[i]} (z = {redshifts[i]:.4f})" for i in closest
    )
    raise ValueError(
        f"No step of this power spectrum is within {REDSHIFT_TOLERANCE} of redshift "
        f"{redshift}. The closest are {candidates}."
    )
