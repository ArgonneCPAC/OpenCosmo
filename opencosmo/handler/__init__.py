# There is some weirdness that can cause things to break if this
# import is not done prior to trying to read data
import hdf5plugin  # type: ignore # noqa: F401

from .handler import OpenCosmoDataHandler
from .im import InMemoryHandler
from .oom import OutOfMemoryHandler

__all__ = ["OpenCosmoDataHandler", "InMemoryHandler", "OutOfMemoryHandler"]
try:
    import mpi4py  # noqa: F401

    from .mpi import MPIHandler  # noqa: F401

    __all__.append("MPIHandler")
except ImportError:
    pass
