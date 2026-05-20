from __future__ import annotations

from typing import TYPE_CHECKING

from opencosmo.io.schema import dataset_schema_length

if TYPE_CHECKING:
    import h5py

    from opencosmo.io.schema import Schema


def allocate(group: h5py.File | h5py.Group, schema: Schema):
    for column_name, column_writer in schema.columns.items():
        if column_writer.shape[0] == 0:
            continue
        group.require_dataset(column_name, column_writer.shape, column_writer.dtype)
    for child_name, child_schema in schema.children.items():
        if dataset_schema_length(child_schema) == 0:
            continue

        child_group = group.require_group(child_name)
        allocate(child_group, child_schema)


def write_columns(group: h5py.File | h5py.Group, schema: Schema):
    for column_path, column_writer in schema.columns.items():
        if column_writer.shape[0] == 0:
            continue
        group[column_path][:] = column_writer.data
        group[column_path].attrs.update(column_writer.attrs)
    for child_name, child_schema in schema.children.items():
        if dataset_schema_length(child_schema) == 0:
            continue
        write_columns(group[child_name], child_schema)


def write_metadata(group: h5py.File | h5py.Group, schema: Schema):
    for path, metadata in schema.attributes.items():
        metadata_group = group.require_group(path)
        metadata_group.attrs.update(metadata)

    for child_name, child_schema in schema.children.items():
        if dataset_schema_length(child_schema) == 0:
            continue
        write_metadata(group[child_name], child_schema)
