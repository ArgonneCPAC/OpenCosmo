"""Generate message models and their application from method signatures."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import NoneType, UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    TypeAliasType,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import astropy.units as u
import numpy as np
from annotated_types import Ge
from pydantic import (
    AfterValidator,
    Field,
    StrictInt,
    TypeAdapter,
    ValidationError,
    create_model,
    model_validator,
)

from opencosmo.column.column import (
    Column,
    ColumnMask,
    CompoundColumnMask,
    ConstructedColumn,
    DerivedScalarValue,
)
from opencosmo.index import DataIndex
from opencosmo.spatial.protocols import Region

from .expression import Expression, ExpressionModel, FiniteNumber, Mask
from .messages.common import NonNegativeInt, normalize_unit
from .messages.region import RegionMessage
from .params import KeywordOnly, VarArgs, VarKwargs
from .rehydrate.expression import expression_to_live, mask_to_live
from .rehydrate.region import region_to_live

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

type ToLive = Callable[[Any], Any]
type MessageValidator = Callable[[Any], Any]


@dataclass(frozen=True)
class WireType:
    """A wire annotation and the conversion of validated wire values to live ones.

    ``to_live`` is ``None`` when the validated wire value is already the live value.
    """

    annotation: object
    to_live: ToLive | None = None


@dataclass(frozen=True)
class BoundParameter:
    """One method parameter and the conversion of its message field.

    For ``*args`` and ``**kwargs`` parameters, ``to_live`` converts each element.
    """

    name: str
    kind: inspect._ParameterKind
    to_live: ToLive | None


@dataclass(frozen=True)
class GeneratedMessage:
    """A message model generated from one method, and how to call that method."""

    method: str
    model: type[ExpressionModel]
    parameters: tuple[BoundParameter, ...]


@dataclass(frozen=True)
class MessageSet:
    """The generated messages for every exposed method of one target type."""

    target: type
    messages: Mapping[str, GeneratedMessage]
    union: object


def __column(expression: Expression) -> Column:
    live = expression_to_live(expression)
    if not isinstance(live, Column):
        raise ValueError("Expression must produce a column")
    return live


def __scalar(expression: Expression) -> DerivedScalarValue:
    live = expression_to_live(expression)
    if not isinstance(live, DerivedScalarValue):
        raise ValueError("Expression must produce a scalar reduction")
    return live


def __rows(rows: tuple[int, ...]) -> np.ndarray:
    return np.asarray(rows, dtype=np.int64)


type UnitString = Annotated[str, AfterValidator(normalize_unit)]
type Keyword = Annotated[str, Field(min_length=1)]

# Live annotations and their wire representations. A whole annotation is looked
# up before it is decomposed, so a specific union can override its members.
WIRE_TYPES: dict[object, WireType] = {
    ColumnMask: WireType(Mask, mask_to_live),
    CompoundColumnMask: WireType(Mask, mask_to_live),
    ConstructedColumn: WireType(Expression, __column),
    DerivedScalarValue: WireType(Expression, __scalar),
    Region: WireType(RegionMessage, region_to_live),
    u.Unit: WireType(UnitString, u.Unit),
    np.ndarray | DataIndex: WireType(tuple[NonNegativeInt, ...], __rows),
    int: WireType(StrictInt),
    float: WireType(FiniteNumber),
    str: WireType(str),
    bool: WireType(bool),
    NoneType: WireType(NoneType),
}

# Live values that the API accepts but that cannot travel over the wire. They
# are dropped from unions; on their own they cannot be generated.
LIVE_ONLY: frozenset[object] = frozenset({np.ndarray, u.Quantity})

# Names that API modules import only for type checking.
NAMESPACE: dict[str, object] = {
    "ColumnMask": ColumnMask,
    "CompoundColumnMask": CompoundColumnMask,
    "ConstructedColumn": ConstructedColumn,
    "DataIndex": DataIndex,
    "DerivedScalarValue": DerivedScalarValue,
    "Ge": Ge,
    "Region": Region,
}


def __lookup(annotation: object) -> WireType | None:
    try:
        return WIRE_TYPES.get(annotation)
    except TypeError:
        return None


def __union(members: list[WireType]) -> WireType:
    annotation: object = Union[tuple(member.annotation for member in members)]  # noqa: UP007
    if all(member.to_live is None for member in members):
        return WireType(annotation)
    adapters: list[TypeAdapter[Any]] = [
        TypeAdapter(member.annotation) for member in members
    ]

    def to_live(value: object) -> object:
        error: ValueError | None = None
        for member, adapter in zip(members, adapters):
            try:
                adapter.validate_python(value, strict=True)
            except ValidationError:
                continue
            if member.to_live is None:
                return value
            try:
                return member.to_live(value)
            except ValueError as member_error:
                error = member_error
        if error is not None:
            raise error
        raise TypeError(f"No wire member accepts {type(value).__name__}")

    return WireType(annotation, to_live)


def __each(to_live: ToLive | None) -> ToLive | None:
    if to_live is None:
        return None
    return lambda values: tuple(to_live(value) for value in values)


def __wire(annotation: object) -> WireType:
    if (known := __lookup(annotation)) is not None:
        return known
    if isinstance(annotation, TypeAliasType):
        return __wire(annotation.__value__)
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        inner = __wire(args[0])
        return WireType(Annotated[(inner.annotation, *args[1:])], inner.to_live)
    if origin is Literal:
        return WireType(annotation)
    if origin in (Union, UnionType):
        members = [__wire(arg) for arg in args if arg not in LIVE_ONLY]
        if not members:
            raise TypeError(f"No member of {annotation!r} can be serialized")
        return __union(members)
    if origin in (Iterable, Sequence, list) or (
        origin is tuple and len(args) == 2 and args[1] is Ellipsis
    ):
        element = __wire(args[0])
        return WireType(
            tuple[element.annotation, ...],  # type: ignore[name-defined]
            __each(element.to_live),
        )
    if origin in (dict, Mapping):
        key, value = __wire(args[0]), __wire(args[1])
        if key.to_live is None and value.to_live is None:
            return WireType(dict[key.annotation, value.annotation])  # type: ignore[name-defined]
        to_key = key.to_live or (lambda item: item)
        to_value = value.to_live or (lambda item: item)
        return WireType(
            dict[key.annotation, value.annotation],  # type: ignore[name-defined]
            lambda mapping: {to_key(k): to_value(v) for k, v in mapping.items()},
        )
    raise TypeError(f"No wire representation for {annotation!r}")


def __field(
    parameter: inspect.Parameter, wire: WireType
) -> tuple[object, object | FieldInfo]:
    default = ... if parameter.default is inspect.Parameter.empty else parameter.default
    match parameter.kind:
        case inspect.Parameter.VAR_POSITIONAL:
            return Annotated[tuple[wire.annotation, ...], VarArgs()], ()  # type: ignore[name-defined]
        case inspect.Parameter.VAR_KEYWORD:
            return (
                Annotated[dict[Keyword, wire.annotation], VarKwargs()],  # type: ignore[name-defined]
                Field(default_factory=dict),
            )
        case inspect.Parameter.KEYWORD_ONLY:
            return Annotated[wire.annotation, KeywordOnly()], default
    return wire.annotation, default


def __generate(
    target: type, method: str, validators: tuple[MessageValidator, ...]
) -> GeneratedMessage:
    function = getattr(target, method)
    hints = get_type_hints(function, localns=NAMESPACE, include_extras=True)
    fields: dict[str, Any] = {"kind": (Literal[method], method)}
    parameters: list[BoundParameter] = []
    for name, parameter in list(inspect.signature(function).parameters.items())[1:]:
        if name not in hints:
            raise TypeError(f"{target.__name__}.{method}({name}) has no annotation")
        try:
            wire = __wire(hints[name])
        except TypeError as error:
            raise TypeError(f"{target.__name__}.{method}({name}): {error}") from error
        fields[name] = __field(parameter, wire)
        parameters.append(BoundParameter(name, parameter.kind, wire.to_live))
    checks = {
        f"check_{i}": model_validator(mode="after")(validator)
        for i, validator in enumerate(validators)
    }
    model = create_model(
        f"{target.__name__}{method.title().replace('_', '')}Message",
        __base__=ExpressionModel,
        __doc__=f"A call to ``{target.__name__}.{method}``.",
        __validators__=checks,  # type: ignore[arg-type]
        **fields,
    )
    model.method = method
    return GeneratedMessage(method, model, tuple(parameters))


def generate_messages(
    target: type,
    methods: Iterable[str],
    validators: Mapping[str, tuple[MessageValidator, ...]] | None = None,
) -> MessageSet:
    """Generate message models for methods of ``target`` from their signatures.

    Parameters
    ----------
    target : type
        The class whose methods the messages represent.
    methods : iterable of str
        The method names to expose.
    validators : mapping, optional
        Additional ``mode="after"`` model validators for each method, for
        constraints that span several parameters.

    Returns
    -------
    MessageSet
        The generated messages and their discriminated union.
    """
    validators = validators or {}
    if unknown := set(validators) - set(methods):
        raise ValueError(f"Validators provided for unexposed methods {sorted(unknown)}")
    messages = {
        method: __generate(target, method, validators.get(method, ()))
        for method in methods
    }
    models = tuple(message.model for message in messages.values())
    union = (
        models[0]
        if len(models) == 1
        else Annotated[Union[models], Field(discriminator="kind")]  # noqa: UP007
    )
    return MessageSet(target, messages, union)


def apply_generated[T](messages: MessageSet, target: T, message: ExpressionModel) -> T:
    """Call the method a generated message represents on ``target``."""
    generated = messages.messages.get(message.method)
    if generated is None or type(message) is not generated.model:
        raise TypeError(
            f"Unsupported {messages.target.__name__} message: {type(message).__name__}"
        )
    args: list[object] = []
    kwargs: dict[str, object] = {}
    for parameter in generated.parameters:
        value = getattr(message, parameter.name)
        convert = parameter.to_live or (lambda item: item)
        match parameter.kind:
            case inspect.Parameter.VAR_POSITIONAL:
                args.extend(convert(item) for item in value)
            case inspect.Parameter.VAR_KEYWORD:
                kwargs.update({key: convert(item) for key, item in value.items()})
            case inspect.Parameter.KEYWORD_ONLY:
                kwargs[parameter.name] = convert(value)
            case _:
                args.append(convert(value))
    return cast("T", getattr(target, generated.method)(*args, **kwargs))


__all__ = [
    "BoundParameter",
    "GeneratedMessage",
    "MessageSet",
    "WireType",
    "apply_generated",
    "generate_messages",
]
