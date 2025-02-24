import hdf5plugin  # noqa
# There is some weirdness that can cause things to break if this
# import is not done prior to trying to read data

from .handler import OpenCosmoDataHandler

__all__ = ["OpenCosmoDataHandler"]
