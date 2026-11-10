from __future__ import annotations

import json
from dataclasses import is_dataclass, replace
from functools import wraps
from inspect import signature
from typing import TYPE_CHECKING, Protocol, cast
from warnings import warn

import numpy as np

from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.io.writer import ColumnCombineStrategy, ColumnWriter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import h5py
    from _typeshed import DataclassInstance
    from mpi4py import MPI
    from pydantic import BaseModel

    from opencosmo.io.schema import Schema


class Loggable(Protocol):
    @property
    def log(self) -> Sequence[BaseModel]: ...


def __is_recording(state: object) -> bool:
    return is_dataclass(state) and getattr(state, "log", None) is not None


def instrument_call(
    serializer: Callable[..., BaseModel], drop: str | Iterable[str] | None = None
):
    drop_columns = {"state"}
    if isinstance(drop, str):
        drop_columns.add(drop)
    elif drop is not None:
        drop_columns = drop_columns.union(drop)

    def make_wrapped_function(func: Callable) -> Callable:

        func_signature = signature(func)
        first_param = next(iter(func_signature.parameters))
        assert first_param == "state"
        assert hasattr(func, "__name__")

        @wraps(func)
        def wrapper[T: DataclassInstance](state: T, *args, log: bool = True, **kwargs):
            # log shares a namespace with the wrapped function's **kwargs, so
            # reject a column or keyword argument that happens to be named log
            if not isinstance(log, bool):
                raise TypeError(
                    "'log' is reserved for instrumentation and must be a bool, "
                    f"got {type(log).__name__}"
                )
            new_state = func(state, *args, **kwargs)
            if not log or not __is_recording(state):
                return new_state

            # bind() gathers *args into a tuple and **kwargs into a dict under
            # their declared parameter names
            bound = func_signature.bind(state, *args, **kwargs)
            bound.apply_defaults()
            arguments = {
                arg: val
                for arg, val in bound.arguments.items()
                if arg not in drop_columns
            }

            return record_call(state, new_state, serializer(**arguments))

        return wrapper

    return make_wrapped_function


def record_call[T: DataclassInstance](state: T, new_state: T, entry: BaseModel) -> T:
    """Return ``new_state`` with ``state``'s log plus ``entry``.

    The log is rebuilt from the input state, so entries recorded by nested
    instrumented calls are discarded in favor of ``entry``.
    """
    if not __is_recording(state):
        return new_state
    log: Sequence[BaseModel] = cast("Loggable", state).log
    return replace(new_state, log=[*log, entry])


def read_log(file: h5py.File | h5py.Group) -> Sequence[BaseModel] | None:
    from opencosmo import Dataset
    from opencosmo.serde import parse_message

    if "header/file/log/entries" not in file:
        return None
    log_list = file["header/file/log/entries"]
    if not log_list:
        return None
    entries = [parse_message(Dataset, json.loads(le)) for le in log_list[:]]
    return entries


def make_log_schema(log: list[BaseModel]):
    if not log:
        # METADATA rather than EMPTY, so ranks with and without history can
        # be compared by the updater instead of failing the schema type check
        return make_schema("log", FileEntry.METADATA, updater=log_schema_updater)
    json_entries = []

    for entry in log:
        json_entries.append(entry.model_dump_json())

    columns = {
        "entries": ColumnWriter.from_numpy_array(
            np.array(json_entries, dtype="S"), ColumnCombineStrategy.EXACT
        )
    }
    return make_schema(
        "log", FileEntry.METADATA, columns=columns, updater=log_schema_updater
    )


def log_schema_updater(schema: Schema, comm: MPI.Comm | None):
    if comm is None:
        return schema

    entries = schema.columns.get("entries")
    entry_column_data = (
        np.array([], dtype="S") if entries is None else entries.get_data(comm)
    )
    all_entries = comm.allgather(entry_column_data)
    for messages in all_entries[1:]:
        if not np.array_equal(messages, all_entries[0]):
            break
    else:
        return schema

    if comm.Get_rank() == 0:
        warn(
            "Found different operation history on different ranks. Operation log will be dropped"
        )

    return schema._replace(columns={})
