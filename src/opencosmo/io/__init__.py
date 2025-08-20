from .io import open, read, write


def missing_package_import_error(name, required):
    raise ValueError(f"Function {name} requires package {required}")


__all__ = ["read", "write", "open", "write_parquet"]


def __getattr__(name):
    if name == "write_parquet":
        try:
            from .parquet import write_parquet
        except ImportError:
            raise ImportError("To use write_parquet you must install pyarrow")
        return write_parquet
    raise ImportError(f"cannot import name '{name}' from 'opencosmo.io'")
