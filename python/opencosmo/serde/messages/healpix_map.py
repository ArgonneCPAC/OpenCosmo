"""Validated messages for HEALPix-map transformations."""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from ..expression import ExpressionModel
from .dataset import (
    DropMessage,
    FilterMessage,
    SelectMessage,
    SortByMessage,
    TakeMessage,
    TakeRangeMessage,
    TakeRowsMessage,
    WithNewColumnsMessage,
)
from .region import ConeRegionMessage, SkyboxRegionMessage

type HealpixBoundRegionMessage = Annotated[
    ConeRegionMessage | SkyboxRegionMessage,
    Field(discriminator="kind"),
]


class HealpixBoundMessage(ExpressionModel):
    """A request to spatially bound a HEALPix map."""

    kind: Literal["healpix_bound"] = "healpix_bound"
    method: ClassVar[str] = "bound"
    region: HealpixBoundRegionMessage
    inclusive: bool = False


type HealpixMapMessage = Annotated[
    FilterMessage
    | SelectMessage
    | DropMessage
    | SortByMessage
    | TakeMessage
    | TakeRangeMessage
    | TakeRowsMessage
    | HealpixBoundMessage
    | WithNewColumnsMessage,
    Field(discriminator="kind"),
]


__all__ = [
    "HealpixBoundMessage",
    "HealpixBoundRegionMessage",
    "HealpixMapMessage",
]
