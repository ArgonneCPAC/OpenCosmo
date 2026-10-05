from __future__ import annotations

from functools import wraps
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Literal, cast

import astropy.units as u  # type: ignore
import numpy as np

from opencosmo.dataset import operations as dsops
from opencosmo.spatial.check import (
    find_coordinate_names_2d,
    find_coordinate_names_3d,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from opencosmo.dataset.dataset import Dataset


def evaluate_to_dataset(
    dataset: Dataset,
    func: Callable,
    vectorize: bool,
    format: str,
    batch_size: int,
    coordinates: Literal["copy"] | list[str] | None,
    allow_overwrite: bool,
    evaluate_kwargs: dict[str, Any],
) -> Dataset:
    """Evaluate a function and construct an in-memory dataset from its outputs."""
    from opencosmo.dataset.build import build_dataset_from_evaluated_data

    state = dataset._state
    baseline_state = dsops.with_units(
        state,
        state.unit_handler.base_convention.value,
        {},
    )
    if coordinates == "copy":
        if state.header.file.is_lightcone:
            coordinate_names = find_coordinate_names_2d(state)
        else:
            coordinate_names = find_coordinate_names_3d(
                state, str(state.header.file.data_type)
            )
    elif coordinates is None:
        coordinate_names = []
    else:
        coordinate_names = coordinates

    source_coordinate_names = set(coordinate_names).intersection(state.columns)
    if source_coordinate_names and len(source_coordinate_names) != len(
        coordinate_names
    ):
        raise ValueError(
            "Coordinate columns must either all be source columns or all be "
            "produced by the evaluation function"
        )
    if source_coordinate_names:
        func = wrap_evaluate_with_coordinates(coordinate_names)(func)

    result = cast(
        "dict[str, np.ndarray | u.Quantity]",
        dsops.evaluate(
            baseline_state,
            func,
            vectorize,
            False,
            format,
            batch_size,
            allow_overwrite,
            **evaluate_kwargs,
        ),
    )
    if coordinate_names and not source_coordinate_names:
        missing_coordinates = set(coordinate_names).difference(result)
        if missing_coordinates:
            raise ValueError(
                "Evaluation function did not produce coordinate columns: "
                f"{sorted(missing_coordinates)}"
            )

    output_length: int | None = None
    for name, output in result.items():
        if not isinstance(output, (np.ndarray, u.Quantity)):
            raise TypeError(
                f"Evaluate output {name!r} must be a NumPy array or Astropy quantity, "
                f"not {type(output).__name__}"
            )
        if output.ndim == 0:
            raise TypeError(
                f"Evaluate output {name!r} must be a NumPy array or Astropy quantity "
                "with a length"
            )
        if output_length is None:
            output_length = len(output)
        elif len(output) != output_length:
            raise ValueError("Evaluate output columns must have equal lengths")

    evaluated = build_dataset_from_evaluated_data(
        result,
        state.header.with_units(state.unit_handler.base_convention),
        coordinate_names=coordinate_names or None,
        max_level=state.tree.max_level if state.tree is not None else None,
    )
    output_state = dsops.with_units(
        evaluated._state,
        state.unit_handler.current_convention.value,
        state.unit_handler.blanket_conversions,
    )
    from opencosmo.dataset.dataset import Dataset

    return Dataset(output_state)


def wrap_evaluate_with_coordinates(coordinate_names: list[str]):
    """Wrap an evaluation function to include its input coordinates in its output."""

    def outer_wrapper(func):
        current_signature = signature(func)
        names = current_signature.parameters.keys()
        uses_data_mapping = "data" in names
        if not uses_data_mapping:
            coordinates_to_add = set(coordinate_names).difference(names)
            new_parameters = list(current_signature.parameters.values()) + [
                Parameter(name, kind=Parameter.KEYWORD_ONLY)
                for name in coordinates_to_add
            ]
            new_signature = current_signature.replace(parameters=new_parameters)
        else:
            coordinates_to_add = set()
            new_signature = current_signature

        @wraps(func)
        def wrapper(*args, **kwargs):
            _ = new_signature.bind(*args, **kwargs)
            if uses_data_mapping:
                data = kwargs["data"]
                coordinate_values = {name: data[name] for name in coordinate_names}
            else:
                coordinate_values = {
                    name: kwargs.pop(name)
                    if name in coordinates_to_add
                    else kwargs[name]
                    for name in coordinate_names
                }

            results = func(*args, **kwargs)
            if not isinstance(results, dict):
                results = {func.__name__: results}
            return __build_output(results, coordinate_values)

        wrapper.__signature__ = new_signature
        return wrapper

    return outer_wrapper


def __build_output(results: dict, coordinates: dict):
    first_output = next(iter(results.values()))
    coordinate_length = len(np.atleast_1d(next(iter(coordinates.values()))))
    has_units = hasattr(next(iter(coordinates.values())), "unit")

    try:
        length = len(first_output)
        if coordinate_length == length:
            return results | coordinates
        if coordinate_length == 1:
            new_coordinates = {
                name: np.full(length, coordinate)
                for name, coordinate in coordinates.items()
            }
            if has_units:
                new_coordinates = {
                    name: coordinate * coordinates[name].unit
                    for name, coordinate in new_coordinates.items()
                }
            return results | new_coordinates
        raise ValueError(
            "The output of the function must be the same length as the coordinates!"
        )
    except TypeError:
        if coordinate_length == 1:
            return results | coordinates
        raise ValueError
