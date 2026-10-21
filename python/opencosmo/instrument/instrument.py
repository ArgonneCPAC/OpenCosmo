from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from functools import wraps
from inspect import signature
from typing import TYPE_CHECKING, Any

from opencosmo.io.schema import FileEntry, Schema

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class LogEntry:
    func_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def instrument_call(func: Callable) -> Callable:
    func_signature = signature(func)
    first_param = next(iter(func_signature.parameters))
    assert first_param == "state"
    assert hasattr(func, "__name__")
    func_name = str(func.__name__)

    @wraps(func)
    def wrapper(state, *args, **kwargs):
        new_state = func(state, *args, **kwargs)
        if not is_dataclass(state) or (log := getattr(state, "log", None)) is None:
            return new_state

        par_names = iter(func_signature.parameters)
        _ = next(par_names)
        kwargs = {name: val for name, val in zip(par_names, args)} | kwargs

        op_log = LogEntry(func_name=func_name, args=(), kwargs=kwargs)

        new_log = log.copy()
        new_log.append(op_log)
        return replace(new_state, log=new_log)

    return wrapper


def make_log_schema(log: list[LogEntry]):
    from opencosmo import __version__

    min_size = len(log) // 10 + 1
    log_children = {}

    for i, entry in enumerate(log):
        group_name = str(i).rjust(min_size, "0")
        attrs = {"func_name": entry.func_name}
        children = {
            "args": make_arg_group_schema(entry.args),
            "kwargs": make_kwarg_group_schema(entry.kwargs),
        }
        entry_schema = Schema(
            name=group_name,
            type=FileEntry.METADATA,
            children=children,
            columns={},
            attributes=attrs,  # type: ignore
        )
        log_children[group_name] = entry_schema

    return Schema(
        name="log",
        type=FileEntry.METADATA,
        children=log_children,
        columns={},
        attributes={"version": __version__},
    )


def make_arg_group_schema(args: tuple[Any, ...]):
    attrs = {str(i): arg for i, arg in enumerate(args)}
    return Schema(
        name="args", type=FileEntry.METADATA, children={}, columns={}, attributes=attrs
    )


def make_kwarg_group_schema(kwargs: dict[str, Any]):
    return Schema(
        name="kwargs",
        type=FileEntry.METADATA,
        children={},
        columns={},
        attributes=kwargs,
    )
