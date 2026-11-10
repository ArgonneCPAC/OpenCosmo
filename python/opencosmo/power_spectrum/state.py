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
# The largest difference between a requested redshift and the step it selects.
# HACC steps sit within a few thousandths of the round redshifts users ask for.
REDSHIFT_TOLERANCE = 0.01


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
        steps=__read_only(group["step"][:]),
        redshifts=__read_only(group["redshift"][:]),
        k=__read_only(group["k"][:]),
        n_modes=__read_only(group["n_modes"][:]),
        components=components,
        unit_handler=unit_handler,
    )


def __read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def with_units(
    state: PowerSpectrumState, convention: str | UnitConvention
) -> PowerSpectrumState:
    convention = UnitConvention(convention)
    if convention == UnitConvention.PHYSICAL:
        raise ValueError(
            "Power spectra are defined in comoving coordinates and have no physical "
            "units. Use the 'comoving' convention for values without factors of h."
        )
    return replace(
        state,
        header=state.header.with_units(convention),
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
        return __redshift_index(state, redshift)
    return None


def __redshift_index(state: PowerSpectrumState, redshift: float) -> int:
    """The index of the step closest to ``redshift``, within REDSHIFT_TOLERANCE."""
    if not np.isfinite(redshift):
        raise ValueError(f"The redshift must be finite, got {redshift}")
    distance = np.abs(state.redshifts - redshift)
    index = int(np.argmin(distance))
    if distance[index] <= REDSHIFT_TOLERANCE:
        return index
    below = np.flatnonzero(state.redshifts < redshift)
    above = np.flatnonzero(state.redshifts > redshift)
    closest = [
        i
        for i in (
            below[np.argmax(state.redshifts[below])] if len(below) else None,
            above[np.argmin(state.redshifts[above])] if len(above) else None,
        )
        if i is not None
    ]
    candidates = " and ".join(
        f"step {state.steps[i]} (z = {state.redshifts[i]:.4f})" for i in closest
    )
    raise ValueError(
        f"No step of this power spectrum is within {REDSHIFT_TOLERANCE} of redshift "
        f"{redshift}. The closest are {candidates}."
    )


def get_k(state: PowerSpectrumState) -> np.ndarray | u.Quantity:
    return state.unit_handler.apply_units({"k": state.k.copy()}, unit_kwargs={})["k"]


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
    return state.unit_handler.apply_units({name: values}, unit_kwargs={})[name]
