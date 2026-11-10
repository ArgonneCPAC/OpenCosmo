from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from opencosmo.units import UnitConvention
from opencosmo.units.handler import make_unit_handler_from_unit_strings

if TYPE_CHECKING:
    import astropy.units as u
    import h5py

    from opencosmo.header import OpenCosmoHeader
    from opencosmo.io.iopen import DatasetTarget
    from opencosmo.units.handler import UnitHandler

PREFIX = "pk_"


@dataclass(frozen=True, eq=False)
class PowerSpectrumState:
    """
    Backend state for a PowerSpectrum. Construct with ``open_state``.

    The axes are small and read at open. The spectra are read from ``group`` when
    they are requested.
    """

    header: OpenCosmoHeader
    group: h5py.Group
    steps: np.ndarray
    redshifts: np.ndarray
    k: np.ndarray
    n_modes: np.ndarray
    components: tuple[str, ...]
    unit_handler: UnitHandler


def open_state(target: DatasetTarget) -> PowerSpectrumState:
    """Open the /power_spectrum group of a discovered power spectrum file."""
    group = target.group["power_spectrum"]
    components = tuple(
        name.removeprefix(PREFIX)
        for name in target.column_names
        if name.startswith(PREFIX)
    )
    unit_handler = make_unit_handler_from_unit_strings(
        target.column_units, target.header, UnitConvention.COMOVING
    )
    return PowerSpectrumState(
        header=target.header.with_units(UnitConvention.COMOVING),
        group=group,
        steps=group["step"][:],
        redshifts=group["redshift"][:],
        k=group["k"][:],
        n_modes=group["n_modes"][:],
        components=components,
        unit_handler=unit_handler,
    )


def with_units(
    state: PowerSpectrumState, convention: str | UnitConvention
) -> PowerSpectrumState:
    convention = UnitConvention(convention)
    # The header has no single redshift, so it cannot be physical. Its comoving
    # values describe every step.
    header_convention = (
        UnitConvention.COMOVING if convention == UnitConvention.PHYSICAL else convention
    )
    return replace(
        state,
        header=state.header.with_units(header_convention),
        unit_handler=state.unit_handler.with_convention(convention),
    )


def step_index(
    state: PowerSpectrumState, step: int | None, redshift: float | None
) -> int | None:
    """The index of the requested step on the step axis, or None for all steps."""
    if step is not None and redshift is not None:
        raise ValueError("Pass either a step or a redshift, not both")
    if step is not None:
        matches = np.flatnonzero(state.steps == step)
        if len(matches) == 0:
            raise ValueError(
                f"Step {step} is not in this power spectrum. Available steps: "
                f"{', '.join(map(str, state.steps))}"
            )
        return int(matches[0])
    if redshift is not None:
        return int(np.argmin(np.abs(state.redshifts - redshift)))
    return None


def scale_factor(state: PowerSpectrumState, index: int | None) -> float | np.ndarray:
    """The scale factor of one step, or of every step as a column."""
    a = 1 / (1 + state.redshifts)
    return a[:, np.newaxis] if index is None else a[index]


def get_k(state: PowerSpectrumState) -> np.ndarray | u.Quantity:
    # Physical wavenumbers depend on the step, so they are returned for every step.
    a = scale_factor(state, None)
    return state.unit_handler.apply_units(
        {"k": state.k}, unit_kwargs={"scale_factor": a}
    )["k"]


def get_spectrum(
    state: PowerSpectrumState, component: str, index: int | None
) -> np.ndarray | u.Quantity:
    if component not in state.components:
        raise ValueError(
            f"This power spectrum has no component {component!r}. Available "
            f"components: {', '.join(state.components)}"
        )
    name = f"{PREFIX}{component}"
    values = state.group[name][:] if index is None else state.group[name][index]
    return state.unit_handler.apply_units(
        {name: values}, unit_kwargs={"scale_factor": scale_factor(state, index)}
    )[name]
