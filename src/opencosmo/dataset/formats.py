from __future__ import annotations

from importlib import import_module

import astropy.units as u
import numpy as np
from astropy.table import QTable


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
            return __convert_to_numpy(data)
        case "pandas":
            return __convert_to_pandas(data)
        case "polars":
            return __convert_to_polars(data)
        case "arrow":
            return __convert_to_arrow(data)
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


def __convert_to_numpy(
    data: dict[str, np.ndarray],
) -> dict[str, np.ndarray] | np.ndarray:
    converted_data = dict(
        map(
            lambda kv: (kv[0], kv[1].value if isinstance(kv[1], u.Quantity) else kv[1]),
            data.items(),
        )
    )
    if len(converted_data) == 1:
        return next(iter(converted_data.values()))
    return converted_data


def __convert_to_pandas(data: dict[str, np.ndarray]):
    import pandas as pd

    numpy_data = __convert_to_numpy(data)
    if isinstance(numpy_data, np.ndarray):  # only one column
        return pd.Series(numpy_data, name=next(iter(data.keys())))

    return pd.DataFrame(numpy_data, copy=True)


def __convert_to_arrow(data: dict[str, np.ndarray]):
    import pyarrow as pa  # type: ignore

    numpy_data = __convert_to_numpy(data)
    if isinstance(numpy_data, np.ndarray):
        return pa.array(numpy_data)

    converted_data = map(
        lambda kv: (kv[0], pa.array(kv[1])),
        data.items(),
    )
    return dict(converted_data)


def __convert_to_polars(data: dict[str, np.ndarray]):
    import polars as pl

    numpy_data = __convert_to_numpy(data)
    if isinstance(numpy_data, np.ndarray):
        return pl.Series(name=next(iter(data.keys())), values=numpy_data)

    return pl.from_dict(data)  # type: ignore
