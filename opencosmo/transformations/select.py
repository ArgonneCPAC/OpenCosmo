from functools import partial
from typing import Optional

from astropy import table

from opencosmo import transformations as t


def select_columns(columns: list[str]) -> t.TableTransformation:
    return partial(__select_columns, columns=columns)


def __select_columns(data: table.Table, columns: list[str]) -> Optional[table.Table]:
    # astropy's error message here is not super great, so I'll make my own

    data_columns = set(data.colnames)
    invalid_columns = set(columns) - data_columns
    if invalid_columns:
        raise ValueError(
            f"Select attempted to get columns that do not exist in the dataset: "
            f"{invalid_columns}"
        )
    return data[*columns]
