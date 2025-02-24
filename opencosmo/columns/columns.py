from astropy.table import Column

from opencosmo import columns


def parse_column_units(column: Column):
    parsers = columns.get_column_parsers()
    for parser in parsers:
        new_column = parser(column)
        if new_column is not None:
            return new_column
    return None
