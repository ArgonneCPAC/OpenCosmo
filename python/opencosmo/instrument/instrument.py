from __future__ import annotations

from dataclasses import is_dataclass, replace
from functools import wraps
from inspect import signature
from typing import TYPE_CHECKING, Any

from opencosmo.io.schema import FileEntry, Schema

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel


def __is_recording(state: object) -> bool:
    return is_dataclass(state) and getattr(state, "log", None) is not None


def instrument_call(serializer: Callable[..., BaseModel]):
    def make_wrapped_function(func: Callable) -> Callable:
        func_signature = signature(func)
        first_param = next(iter(func_signature.parameters))
        assert first_param == "state"
        assert hasattr(func, "__name__")

        @wraps(func)
        def wrapper(state, *args, log: bool = True, **kwargs):
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


def record_call[T](state: T, new_state: T, entry: BaseModel) -> T:
    """Return ``new_state`` with ``state``'s log plus ``entry``.

    The log is rebuilt from the input state, so entries recorded by nested
    instrumented calls are discarded in favor of ``entry``.
    """
    if not __is_recording(state):
        return new_state
    return replace(new_state, log=[*state.log, entry])  # type: ignore[attr-defined, type-var]


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
