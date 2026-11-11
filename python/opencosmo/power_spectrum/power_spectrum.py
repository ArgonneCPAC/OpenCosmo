from __future__ import annotations

from collections.abc import Iterable
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


"""
Note to maintainers: The PowerSpectrum class intentionally does not mirror the state/facade
construction of the other user-facing classes. It does not holds datasets, implements only
a small subset of the standard API, and reads all of its data eagerly. In general, datat types 
like this should be treated separately.
"""


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
        pk_cdm = ps.select("cdm").with_redshifts(0).get_data()
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
    # _spectra Shape (n_components, n_steps, n_k), in the order of ``components``.
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

    def select(self, *components: str | Iterable[str]) -> PowerSpectrum:
        """
        Create a new power spectrum with a subset of the components, like
        :py:meth:`Dataset.select <opencosmo.Dataset.select>`. Pass "all" to keep
        every component.

        Parameters
        ----------
        *components : str or list[str]
            The components to keep: "all", one of :py:attr:`components`, or
            several of them.

        Returns
        -------
        power_spectrum : PowerSpectrum
            The new power spectrum with only the selected components.

        Raises
        ------
        ValueError
            If any of the given components are not in the power spectrum.
        """
        names = _resolve_components(self.components, _flatten_args(components))
        indices = [self.components.index(name) for name in names]
        return replace(
            self,
            components=tuple(names),
            _spectra=_read_only(self._spectra[indices]),
        )

    def drop(self, *components: str | Iterable[str]) -> PowerSpectrum:
        """
        Create a new power spectrum without the provided components, like
        :py:meth:`Dataset.drop <opencosmo.Dataset.drop>`.

        Parameters
        ----------
        *components : str or list[str]
            The components to drop: one of :py:attr:`components` or several of
            them.

        Returns
        -------
        power_spectrum : PowerSpectrum
            The new power spectrum without the dropped components.

        Raises
        ------
        ValueError
            If any of the given components are not in the power spectrum, or if
            dropping them would leave no components.
        """
        requested = _flatten_args(components)
        if requested == ["all"]:
            requested = list(self.components)
        dropped = set(_resolve_components(self.components, requested))
        kept = tuple(name for name in self.components if name not in dropped)
        if not kept:
            raise ValueError("Cannot drop every component of a power spectrum")
        indices = [self.components.index(name) for name in kept]
        return replace(
            self, components=kept, _spectra=_read_only(self._spectra[indices])
        )

    def with_steps(self, *steps: int | Iterable[int]) -> PowerSpectrum:
        """
        Create a new power spectrum that covers only the given simulation steps.

        Parameters
        ----------
        *steps : int or list[int]
            The steps to keep. Each must be one of :py:attr:`steps`.

        Returns
        -------
        power_spectrum : PowerSpectrum
            The new power spectrum restricted to the selected steps.

        Raises
        ------
        ValueError
            If any step is not in the power spectrum.
        """
        return _with_step_indices(
            self, [_step_index(self.steps, step) for step in _flatten_args(steps)]
        )

    def with_redshifts(self, *redshifts: float | Iterable[float]) -> PowerSpectrum:
        """
        Create a new power spectrum that covers only the steps closest to the
        given redshifts. Each value selects the step whose redshift in
        :py:attr:`redshifts` is closest to it, and it fails if that step is more
        than 0.01 away. HACC steps lie within a few thousandths of round
        redshifts, so ``with_redshifts(0.5)`` selects the step at z = 0.4987 in
        the SCIDAC simulations.

        Parameters
        ----------
        *redshifts : float or list[float]
            The redshifts whose closest steps should be kept.

        Returns
        -------
        power_spectrum : PowerSpectrum
            The new power spectrum restricted to the selected steps.

        Raises
        ------
        ValueError
            If no step is within 0.01 of a redshift (the message names the
            closest steps), or if a redshift is not finite.
        """
        return _with_step_indices(
            self,
            [
                _redshift_index(self.steps, self.redshifts, z)
                for z in _flatten_args(redshifts)
            ],
        )

    def get_data(
        self,
        format: str = "astropy",
        return_error: bool = False,
    ):
        """
        Get the power spectra of all :py:attr:`components` at all
        :py:attr:`steps`, as arrays of shape (n_steps, n_k). Use
        :py:meth:`select`/:py:meth:`drop` to choose components and
        :py:meth:`with_steps`/:py:meth:`with_redshifts` to choose steps first.

        When more than one step is selected, the output also has a "redshift"
        column with the redshift of each step, with shape (n_steps,). The
        spectrum of a component is NaN at the steps where it has no particles,
        for example the black holes at early steps.

        Parameters
        ----------
        return_error: bool, default = False
            Return the sampling error associated with each component: P(k) / sqrt(N_modes(k))
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
            If the format is unknown.
        ImportError
            If the format requires a package that is not installed.
        """
        from opencosmo.dataset.formats import convert_data, verify_format

        data = {}
        for name in self.components:
            spectrum = self._spectra[self.components.index(name)]
            # The stored arrays are read-only, so slices and views of them can be
            # handed out without copying.
            data[f"{PREFIX}{name}"] = spectrum
        data = self._unit_handler.apply_units(data, unit_kwargs={})
        data = {name.removeprefix(PREFIX): value for name, value in data.items()}
        if return_error:
            data |= compute_error(data, self.n_modes)
        if len(self.steps) > 1:
            data["redshift"] = self.redshifts

        verify_format(format)
        return convert_data(data, format, wrap_single=False)

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
        steps=_read_only(group["step"][:]),
        redshifts=_read_only(group["redshift"][:]),
        _k=_read_only(group["k"][:]),
        n_modes=_read_only(group["n_modes"][:]),
        components=tuple(name.removeprefix(PREFIX) for name in names),
        _spectra=_read_only(np.stack([group[name][:] for name in names])),
        _unit_handler=unit_handler,
    )


def _read_only(array: np.ndarray) -> np.ndarray:
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
        requested = list(available) if components == ["all"] else list(components)
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


def compute_error(data: dict[str, np.ndarray], nmodes: np.ndarray):
    return {f"{name}_error": d / np.sqrt(nmodes) for name, d in data.items()}


def _flatten_args(values: tuple) -> list:
    """Flatten ``select("a", ["b", "c"])``-style variadic arguments."""
    flat = []
    for value in values:
        if isinstance(value, str) or not isinstance(value, Iterable):
            flat.append(value)
        else:
            flat.extend(value)
    return flat


def _with_step_indices(power_spectrum: PowerSpectrum, indices: list[int]):
    return replace(
        power_spectrum,
        steps=power_spectrum.steps[indices],
        redshifts=power_spectrum.redshifts[indices],
        _spectra=_read_only(power_spectrum._spectra[:, indices, :]),
    )


def _step_index(steps: np.ndarray, step: int) -> int:
    """The index of ``step`` in ``steps``."""
    matches = np.flatnonzero(steps == step)
    if len(matches) == 0:
        raise ValueError(
            f"Step {step} is not in this power spectrum. Available steps: "
            f"{', '.join(map(str, steps))}"
        )
    return int(matches[0])


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
