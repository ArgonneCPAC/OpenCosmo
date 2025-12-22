from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import h5py

    from opencosmo.io.verify import ColumnWriter


def allocate(file: h5py.File, writers: dict[str, ColumnWriter]):
    for column_path, column_writer in writers.items():
        file.require_dataset(column_path, len(column_writer), column_writer.dtype)


def write_columns(file: h5py.File, writers: dict[str, ColumnWriter]):
    for column_path, column_writer in writers.items():
        file[column_path][:] = column_writer.data
        file[column_path].attrs.update(column_writer.attrs)


def write_metadata(file: h5py.File, metadata: dict[str, dict[str, Any]]):
    for path, data in metadata.items():
        group = file.require_group(path)
        group.attrs.update(data)
