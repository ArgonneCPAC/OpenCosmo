from __future__ import annotations

import json
from dataclasses import is_dataclass, replace
from functools import wraps
from inspect import signature
from typing import TYPE_CHECKING, Protocol, TypeGuard

import numpy as np

from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.io.writer import ColumnWriter

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import h5py
    from _typeshed import DataclassInstance
    from pydantic import BaseModel


class Loggable(Protocol):
    @property
    def log(self) -> Sequence[BaseModel]: ...


def __is_recording(state: object) -> TypeGuard[Loggable]:
    return is_dataclass(state) and getattr(state, "log", None) is not None


def instrument_call(serializer: Callable[..., BaseModel]):
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
            arguments = bound.arguments
            del arguments["state"]
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
    return replace(new_state, log=[*state.log, entry])  # type: ignore[attr-defined, type-var]


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
        return make_schema("log", FileEntry.EMPTY)
    json_entries = []

    for entry in log:
        json_entries.append(entry.model_dump_json())

    columns = {
        "entries": ColumnWriter.from_numpy_array(np.array(json_entries, dtype="S"))
    }
    return make_schema("log", FileEntry.METADATA, columns=columns)
