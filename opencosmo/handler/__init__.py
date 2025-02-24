# There is some weirdness that can cause things to break if this
# import is not done prior to trying to read data
import hdf5plugin  # noqa # type: ignore

from .handler import OpenCosmoDataHandler
from .im import InMemoryHandler

__all__ = ["OpenCosmoDataHandler", "InMemoryHandler"]
