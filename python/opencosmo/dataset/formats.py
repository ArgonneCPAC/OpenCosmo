from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Iterable

import astropy.units as u
import numpy as np
from astropy.table import Column, QTable

if TYPE_CHECKING:
    from opencosmo import Dataset


def verify_format(output_format: str):
    match output_format:
        case "astropy":
            return
        case "numpy":  # these two are core dependencies
            return
        case "pandas":
            import_name = "pandas"
        case "arrow":
            import_name = "pyarrow"
        case "polars":
            import_name = "polars"
        case "jax":
            import_name = "jax"
        case _:
            raise ValueError(f"Unknown data output format {output_format}")

    __verify_import(import_name, output_format)


def __verify_import(import_name: str, format_name: str):
    try:
        import_module(import_name)
    except ImportError as e:
        raise ImportError(
            f"Data was requested in {format_name} format but could not import {import_name} package. Got '{e}'"
        )


def convert_data(data: dict[str, np.ndarray], output_format: str):
    match output_format:
        case "astropy":
            return __convert_to_astropy(data)
        case "numpy":
            return convert_to_numpy(data)
        case "pandas":
            return __convert_to_pandas(data)
        case "polars":
            return __convert_to_polars(data)
        case "arrow":
            return __convert_to_arrow(data)
        case "jax":
            return __convert_to_jax(data)
        case _:
            raise ValueError(f"Unknown data output format {output_format}")


def fetch_as_dict(
    dataset: Dataset,
    requires_names: Iterable[str],
    output_format: str,
    unpack: bool = True,
) -> dict[str, Any]:
    """
    Fetch the requested columns and return them as a {name: container} dict in
    the user's requested format. Routes through astropy so that Quantities (with
    units) survive into the conversion step; other formats then receive plain
    values via to_format_dict.
    """
    requires_names = list(requires_names)
    raw = dataset.select(requires_names).get_data(format="astropy", unpack=unpack)
    if isinstance(raw, QTable):
        raw = {name: raw[name] for name in raw.colnames}
    elif not isinstance(raw, dict):
        raw = {requires_names[0]: raw}
    if output_format == "astropy":
        return raw
    return to_format_dict(raw, output_format)


def to_format_dict(data: dict[str, np.ndarray], output_format: str) -> dict:
    """
    Convert each column of a numpy/astropy dict to the requested format,
    preserving the dict shape. Unlike convert_data, this never wraps the
    result in a higher-level container (DataFrame, QTable, ...). Used to
    feed user-supplied evaluate functions when the upstream data is
    numpy-shaped (e.g. when reading from the column cache).
    """
    if output_format == "astropy":
        return data

    def strip(value):
        return value.value if isinstance(value, (u.Quantity, Column)) else value

    match output_format:
        case "numpy":
            return {k: strip(v) for k, v in data.items()}
        case "jax":
            import jax.numpy as jnp

            return {k: jnp.asarray(strip(v)) for k, v in data.items()}
        case "pandas":
            import pandas as pd

            return {k: pd.Series(strip(v)) for k, v in data.items()}
        case "polars":
            import polars as pl

            return {k: pl.Series(values=strip(v)) for k, v in data.items()}
        case "arrow":
            import pyarrow as pa  # type: ignore

            return {k: pa.array(strip(v)) for k, v in data.items()}
        case _:
            raise ValueError(f"Unknown data output format {output_format}")


def to_numpy_dict(data: dict) -> dict[str, np.ndarray]:
    """
    Convert each value in a dict-of-format-arrays back to a numpy array
    suitable for the column cache. Astropy Quantities are preserved so
    that downstream unit handling continues to work; other formats are
    converted to plain numpy with no unit information.
    """
    result: dict[str, np.ndarray] = {}
    for name, value in data.items():
        if isinstance(value, (u.Quantity, np.ndarray)):
            result[name] = value
        else:
            result[name] = np.asarray(value)
    return result


def stack_rows(values: list, output_format: str):
    """
    Stack a list of per-row values into a 1-D container in the target format.
    Used by row-wise evaluation strategies to assemble output without
    preallocation, which would break for formats with immutable arrays
    (e.g. jax).
    """
    match output_format:
        case "astropy":
            if values and isinstance(values[0], u.Quantity):
                return u.Quantity(values)
            return np.array(values)
        case "numpy":
            return np.array(values)
        case "jax":
            import jax.numpy as jnp

            return jnp.array(values)
        case "pandas":
            import pandas as pd

            return pd.Series(values)
        case "polars":
            import polars as pl

            return pl.Series(values=values)
        case "arrow":
            import pyarrow as pa  # type: ignore

            return pa.array(values)
        case _:
            raise ValueError(f"Unknown data output format {output_format}")


def concat_chunks(chunks: list, output_format: str):
    """
    Concatenate a list of per-chunk arrays into a single container in the
    target format.
    """
    match output_format:
        case "astropy" | "numpy":
            return np.concatenate(chunks)
        case "jax":
            import jax.numpy as jnp

            return jnp.concatenate(chunks)
        case "pandas":
            import pandas as pd

            return pd.concat(chunks, ignore_index=True)
        case "polars":
            import polars as pl

            return pl.concat(chunks)
        case "arrow":
            import pyarrow as pa  # type: ignore

            return pa.concat_arrays(chunks)
        case _:
            raise ValueError(f"Unknown data output format {output_format}")


def __convert_to_astropy(data: dict[str, np.ndarray]) -> QTable:
    if len(data) == 1:
        return next(iter(data.values()))
    if any(
        (isinstance(d, u.Quantity) and d.isscalar) or not isinstance(d, np.ndarray)
        for d in data.values()
    ):
        return data

    return QTable(data, copy=False)


def convert_to_numpy(
    data: dict[str, np.ndarray],
) -> dict[str, np.ndarray] | np.ndarray:
    converted_data = dict(
        map(
            lambda kv: (
                kv[0],
                kv[1].value if isinstance(kv[1], (u.Quantity, Column)) else kv[1],
            ),
            data.items(),
        )
    )
    if len(converted_data) == 1:
        return next(iter(converted_data.values()))
    return converted_data


def __convert_to_pandas(data: dict[str, np.ndarray]):
    import pandas as pd

    numpy_data = convert_to_numpy(data)
    if isinstance(numpy_data, np.ndarray):  # only one column
        return pd.Series(numpy_data, name=next(iter(data.keys())))

    return pd.DataFrame(numpy_data, copy=True)


def __convert_to_arrow(data: dict[str, np.ndarray]):
    import pyarrow as pa  # type: ignore

    numpy_data = convert_to_numpy(data)
    if isinstance(numpy_data, np.ndarray):
        return pa.array(numpy_data)

    converted_data = map(
        lambda kv: (kv[0], pa.array(kv[1])),
        data.items(),
    )
    return dict(converted_data)


def __convert_to_polars(data: dict[str, np.ndarray]):
    import polars as pl

    numpy_data = convert_to_numpy(data)
    if isinstance(numpy_data, np.ndarray):
        return pl.Series(name=next(iter(data.keys())), values=numpy_data)

    return pl.from_dict(data)  # type: ignore


def __convert_to_jax(data: dict[str, np.ndarray]):
    import jax.numpy as jnp

    output_data = convert_to_numpy(data)
    if isinstance(output_data, np.ndarray):
        return jnp.asarray(output_data)
    return {key: jnp.asarray(value) for key, value in output_data.items()}
