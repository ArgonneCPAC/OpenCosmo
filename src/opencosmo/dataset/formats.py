from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import astropy.units as u
from astropy.table import QTable

if TYPE_CHECKING:
    import numpy as np


def verify_format(format: str):
    match format:
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
            raise ValueError(f"Unknown data format {format}")

    __verify_import(import_name, format)


def __verify_import(import_name: str, format_name: str):
    try:
        import_module(import_name)
    except ImportError as e:
        raise ImportError(
            f"Data was requested in {format_name} format but could not import {import_name} package. Got '{e}'"
        )


def convert_data(data: dict[str, np.ndarray], format: str):
    match format:
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
            raise ValueError(f"Unknown data format {format}")


def __convert_to_astropy(data: dict[str, np.ndarray]) -> QTable:
    return QTable(data, copy=False)


def __convert_to_numpy(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    converted_data = map(
        lambda kv: (kv[0], kv[1].value if isinstance(kv[1], u.Quantity) else kv[1]),
        data.items(),
    )
    return dict(converted_data)


def __convert_to_pandas(data: dict[str, np.ndarray]):
    import pandas as pd

    numpy_data = __convert_to_numpy(data)
    return pd.DataFrame(numpy_data, copy=True)


def __convert_to_arrow(data: dict[str, np.ndarray]):
    import pyarrow as pa

    numpy_data = __convert_to_numpy(data)
    converted_data = map(
        lambda kv: (kv[0], pa.array(kv[1])),
        data.items(),
    )
    return dict(converted_data)


def __convert_to_polars(data: dict[str, np.ndarray]):
    import polars as pl

    data = __convert_to_numpy(data)
    return pl.from_dict(data)
