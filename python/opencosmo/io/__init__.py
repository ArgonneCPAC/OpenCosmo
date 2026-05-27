import hdf5plugin  # type: ignore # noqa: F401

from .io import open, write


def missing_package_import_error(name, required):
    raise ValueError(f"Function {name} requires package {required}")


__all__ = ["write", "open", "write_parquet"]  # noqa: F822


def __getattr__(name):
    if name == "write_parquet":
        try:
            from .parquet import write_parquet
        except ImportError:
            raise ImportError("To use write_parquet you must install pyarrow")
        return write_parquet
    raise ImportError(f"cannot import name '{name}' from 'opencosmo.io'")
