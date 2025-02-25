import numpy as np
from astropy.table import Column, Table, join

from opencosmo import transformations


def apply_table_transformations(
    table: Table, transformations: list[transformations.TableTransformations]
):
    output_table = table
    for t in transformations:
        if (new_table := t.transform(table)) is not None:
            output_table = combine_tables(table, new_table)
    return output_table


def apply_filter_transformations(
    table: Table, transformations: list[transformations.FilterTransformations]
):
    mask = np.ones(len(table), dtype=bool)
    for t in transformations:
        if (new_mask := t.transform(table)) is not None:
            mask &= new_mask
    return table[mask]


def apply_column_transformations(
    table: Table, transformations: list[transformations.ColumnTransformations]
):
    for t in transformations:
        column_name = t.column_name
        if column_name not in table.columns:
            raise ValueError(f"Column {column_name} not found in table")
        column = table[column_name]
        if (new_column := t.transform(column)) is not None:
            table[column_name] = new_column
    return table


def combine_tables(table: Table, new_table: Table):
    if len(table) != len(new_table):
        raise ValueError("Tables must have the same length")
    original_columns = set(table.columns)
    updated_columns = set(new_table.columns)

    new_columns = updated_columns - original_columns
    modified_columns = updated_columns & original_columns

    for column in modified_columns:
        table[column] = new_table[column]

    new_table = table[list(new_columns)]
    return join(table, new_table)
