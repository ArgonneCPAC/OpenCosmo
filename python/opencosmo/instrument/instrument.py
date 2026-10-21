from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from functools import wraps
from inspect import signature
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class Log:
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

        op_log = Log(func_name=func_name, args=args, kwargs=kwargs)

        new_log = log.copy()
        new_log.append(op_log)
        return replace(new_state, log=new_log)

    return wrapper
