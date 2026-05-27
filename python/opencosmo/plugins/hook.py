from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, TypeVar

Ctx = TypeVar("Ctx")
R = TypeVar("R")

type Predicate[Ctx] = Callable[[Ctx], bool]


@dataclass(frozen=True)
class Registration[Ctx]:
    predicate: Predicate[Ctx]
    transform: Callable[[Ctx], Any]


_registry: dict[str, list[Registration]] = defaultdict(list)


def hook(name: str, *, when: Predicate = lambda _: True):
    """Decorator that registers a function as a hook implementation.

    The decorated function receives and returns a context dataclass. For fold
    hooks, it should return a (possibly modified) copy of the context; use
    dataclasses.replace() to do so functionally. For query hooks, it should
    return either a result value or None to signal no match.

    Parameters
    ----------
    name:
        The hook point name. Use a constant from HookPoint.
    when:
        Predicate called with the context. The hook only fires when this
        returns True.
    """

    def decorator(fn: Callable) -> Callable:
        _registry[name].append(Registration(when, fn))
        return fn

    return decorator


def fold(name: str, ctx: Ctx) -> Ctx:
    """Apply all registered hooks for *name* in order, threading ctx through.

    Each matching hook receives the output of the previous one. Non-matching
    hooks (predicate returns False) are skipped and ctx passes through unchanged.
    """
    return reduce(
        lambda c, reg: reg.transform(c) if reg.predicate(c) else c,
        _registry[name],
        ctx,
    )


def query(name: str, ctx: Any) -> Any | None:
    """Return the result of the first matching hook for *name*, or None.

    Intended for hook points where at most one plugin should respond — the
    first hook whose predicate matches is called, and its return value is
    returned immediately. Subsequent hooks are not evaluated.
    """
    return next(
        (
            result
            for reg in _registry[name]
            if reg.predicate(ctx) and (result := reg.transform(ctx)) is not None
        ),
        None,
    )
