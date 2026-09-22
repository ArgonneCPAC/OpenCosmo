from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator

import click

if TYPE_CHECKING:
    from opencosmo.io.discover import FileLayout

from opencosmo.uuid import get_string_uuid

logger = logging.getLogger(__name__)

LAYOUT_VERSION = 1
CACHE_DISABLED = os.environ.get("OPENCOSMO_DISABLE_CACHE", "") not in ("", "0")
SHARED_CACHE_DIRNAME = ".opencosmo"


def __user_cache_dir() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        cache_root = home / "Library" / "Caches"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME", home / ".cache")
        cache_root = Path(xdg_cache)

    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_root / "opencosmo"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


@contextlib.contextmanager
def _sqlite_connection(
    cache_db_path: Path, *, readonly: bool
) -> Iterator[sqlite3.Connection | None]:
    try:
        if readonly:
            if not cache_db_path.exists():
                yield None
                return
            try:
                conn = sqlite3.connect(
                    f"file:{cache_db_path}?mode=ro",
                    uri=True,
                    timeout=30.0,
                )
            except sqlite3.OperationalError:
                # A WAL-mode database in a read-only directory cannot be opened
                # even for reads, because SQLite wants to create -wal/-shm
                # sidecars. immutable=1 skips that at the cost of assuming no
                # concurrent writer, which holds for a published shared cache.
                conn = sqlite3.connect(
                    f"file:{cache_db_path}?immutable=1",
                    uri=True,
                    timeout=30.0,
                )
        else:
            conn = sqlite3.connect(cache_db_path, timeout=30.0)
            # WAL is unreliable on some network filesystems; treat write failure as non-fatal.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
    except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError):
        logger.debug("sqlite connection failed", exc_info=True)
        yield None
        return

    # Outside the try above: a generator must yield exactly once, so an exception
    # raised by the caller's body must not land on the `yield None` fallback.
    try:
        yield conn
    finally:
        try:
            conn.commit()
        except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError):
            logger.debug("sqlite commit failed", exc_info=True)
        finally:
            conn.close()


@contextlib.contextmanager
def open_cache_db_for_read(cache_dir: Path) -> Iterator[sqlite3.Connection | None]:
    cache_path = cache_dir / "files.db"
    with _sqlite_connection(cache_path, readonly=True) as conn:
        yield conn


@contextlib.contextmanager
def open_cache_db_for_write(cache_dir: Path) -> Iterator[sqlite3.Connection | None]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "files.db"
    with _sqlite_connection(cache_path, readonly=False) as conn:
        yield conn


def __ensure_cache_tables(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            path STRING PRIMARY KEY,
            mtime REAL NOT NULL,
            layout_version INTEGER NOT NULL
        )
        """
    )


def get_directory_read_cache_dir(directory: Path) -> Path:
    """Resolve the cache directory used for *reads* from the given data dir."""
    if not directory.is_dir():
        raise NotADirectoryError(
            f"Expected a directory for cache resolution: {directory}"
        )

    shared_cache_path = directory / SHARED_CACHE_DIRNAME
    shared_db_path = shared_cache_path / "files.db"
    if shared_db_path.is_file():
        return shared_cache_path

    return __user_cache_dir()


def get_directory_write_cache_dir(directory: Path) -> Path:
    """Resolve the cache directory used for *writes* from the given data dir."""
    if not directory.is_dir():
        raise NotADirectoryError(
            f"Expected a directory for cache resolution: {directory}"
        )
    return __user_cache_dir()


def __relative_cache_root(cache_dir: Path) -> Path | None:
    """
    Return the data directory a shared cache is anchored to, or None.

    Relative keys are only meaningful for a directory-local shared cache, where
    the data files live alongside the ``.opencosmo`` directory holding the
    SQLite database. The per-user cache has no such anchor.
    """
    if cache_dir.name != SHARED_CACHE_DIRNAME:
        return None
    return cache_dir.parent


def build_cache_key(path: Path, root: Path | None) -> str:
    """
    Build the string used to key a file in the SQLite index and name its blob.

    ``root`` is the directory the key is taken relative to, or None for an
    absolute key.
    """
    if root is None:
        return str(path)
    return str(path.relative_to(root))


def __candidate_cache_keys(cache_dir: Path, path: Path) -> tuple[str, ...]:
    """
    Keys a file may be stored under, in order of preference.

    A shared cache may have been populated either with absolute paths or with
    paths relative to the data directory, so reads have to consider both.
    """
    keys = [str(path)]
    root = __relative_cache_root(cache_dir)
    if root is not None:
        try:
            keys.append(build_cache_key(path, root))
        except ValueError:
            pass
    return tuple(keys)


def sort_files_by_cache_dir(file_paths: Iterable[Path]) -> dict[Path, list[Path]]:
    """Group file paths by the cache directory used for *reads*."""

    files_by_dir: defaultdict[Path, list[Path]] = defaultdict(list)
    for file_path in file_paths:
        files_by_dir[file_path.parent].append(file_path)

    files_by_cache_dir: defaultdict[Path, list[Path]] = defaultdict(list)
    files_by_dir_dict: dict[Path, list[Path]] = dict(files_by_dir)
    for dir_path, paths in files_by_dir_dict.items():
        cache_dir = get_directory_read_cache_dir(dir_path)
        files_by_cache_dir[cache_dir].extend(paths)
    return dict(files_by_cache_dir)


def cache_layouts(layouts: list[FileLayout]) -> None:
    if CACHE_DISABLED:
        return

    layouts_by_path = {layout.path: layout for layout in layouts}
    files_by_cache_dir: defaultdict[Path, list[Path]] = defaultdict(list)
    for path in layouts_by_path.keys():
        cache_dir = get_directory_write_cache_dir(path.parent)
        files_by_cache_dir[cache_dir].append(path)

    layouts_by_cache_dir: dict[Path, list[FileLayout]] = {}
    for cache_dir, cache_paths in files_by_cache_dir.items():
        layouts_by_cache_dir[cache_dir] = [
            layouts_by_path[path] for path in cache_paths
        ]

    for cache_dir, cache_layouts_ in layouts_by_cache_dir.items():
        write_layouts(cache_dir, cache_layouts_)


def write_layouts(
    cache_dir: Path, layouts: list[FileLayout], root: Path | None = None
) -> None:
    """
    Write layouts and their SQLite index entries into ``cache_dir``.

    When ``root`` is given, files are keyed by their path relative to it rather
    than by their absolute path, so the cache stays valid when the same data
    directory is mounted somewhere else.
    """
    if not layouts:
        return

    os.makedirs(cache_dir, exist_ok=True)
    db_entries: list[dict[str, object]] = []
    for layout in layouts:
        if layout.error is not None:
            continue
        key = build_cache_key(layout.path, root)
        uuid = get_string_uuid(key)
        from opencosmo.io import discover

        blob = discover.encode_file_layout_blob(layout)
        final_blob_path = cache_dir / f"{uuid}.json"
        tmp_blob_path = cache_dir / f"{uuid}.json.tmp.{os.getpid()}"
        with open(tmp_blob_path, "w", encoding="utf-8") as f:
            json.dump(blob, f)
        os.replace(tmp_blob_path, final_blob_path)
        db_entries.append(build_db_entry(layout.path, key))

    if not db_entries:
        return

    query = """
        INSERT OR REPLACE INTO files (path, mtime, layout_version)
        VALUES(:path, :mtime, :layout_version)
    """

    with open_cache_db_for_write(cache_dir) as conn:
        if conn is None:
            return
        __ensure_cache_tables(conn)
        cursor = conn.cursor()
        cursor.executemany(query, db_entries)


@click.command(name="cache-layouts")
@click.argument(
    "directory",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
    required=True,
)
@click.option("--pattern", type=str, required=False, default="*.hdf5")
@click.option("--relative", "-r", is_flag=True)
def cache_layouts_cli(directory: Path, pattern: str, relative: bool = False):
    return populate_directory_cache(directory, pattern=pattern, relative_path=relative)


def populate_directory_cache(
    directory: Path, *, pattern: str = "*.hdf5", relative_path: bool = False
) -> PopulateResult:
    """
    Build layouts for every matching file in ``directory`` and write a shared cache.

    This is an administrative tool for pre-populating the shared cache on HPC
    systems, where the data directory is later exposed read-only to users. It
    writes to ``directory/.opencosmo``, which is the location
    :func:`get_directory_read_cache_dir` prefers when the data directory is not
    writable. This deliberately bypasses :func:`get_directory_write_cache_dir`
    (which always resolves to the per-user cache) and is the only supported way
    to produce a shared cache.

    Unlike the implicit cache path used by ``discover_all``, failures here are
    reported rather than silently swallowed, and files that fail to discover are
    skipped rather than cached.

    Parameters
    ----------
    directory : Path
        Directory of OpenCosmo HDF5 files to index. Not searched recursively.
    pattern : str, default "*.hdf5"
        Glob pattern selecting which files to index.
    relative_path : bool, default False
        Key cached files by their path relative to ``directory`` rather than by
        their absolute path. Use this when the cache travels with the data and
        the data is mounted at different absolute paths by different readers,
        such as containerized query backends. Reads accept either form, so a
        relative cache does not need any reader configuration.

    Returns
    -------
    PopulateResult
        Counts of cached and skipped files, the failures by path, and the cache
        directory that was written.

    Raises
    ------
    NotADirectoryError
        If ``directory`` does not exist or is not a directory.
    OSError
        If the shared cache directory cannot be created or written.
    """
    from opencosmo.io.discover import discover_file

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    root = directory.resolve()
    cache_dir = root / SHARED_CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p.resolve() for p in root.glob(pattern) if p.is_file())

    layouts: list[FileLayout] = []
    failures: dict[Path, str] = {}
    for path in paths:
        layout = discover_file(path)
        if layout.error is not None:
            failures[path] = layout.error
            continue
        layouts.append(layout)

    write_layouts(cache_dir, layouts, root if relative_path else None)
    __finalize_shared_db(cache_dir)

    return PopulateResult(
        cache_dir=cache_dir,
        cached=tuple(layout.path for layout in layouts),
        failed=failures,
    )


def __finalize_shared_db(cache_dir: Path) -> None:
    """
    Checkpoint and leave the shared DB in a form readable from a read-only dir.

    ``write_layouts`` puts the database in WAL mode, which is right for the
    concurrent per-user cache but fatal for a published shared cache: SQLite
    cannot open a WAL database at all, even read-only, without being able to
    create ``-wal``/``-shm`` sidecars next to it. Switching back to the rollback
    journal removes them.
    """
    db_path = cache_dir / "files.db"
    if not db_path.exists():
        return
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.commit()
    finally:
        conn.close()


@dataclass(frozen=True)
class PopulateResult:
    """Summary of a :func:`populate_directory_cache` run."""

    cache_dir: Path
    """Directory the shared cache was written to."""

    cached: tuple[Path, ...]
    """Paths successfully discovered and cached, sorted."""

    failed: dict[Path, str]
    """Paths that failed discovery, mapped to the discovery error message."""


def build_db_entry(path: Path, key: str | None = None) -> dict[str, object]:
    # Any change to FileLayout/GroupLayout/LinkLayout fields or to
    # encode_file_layout_blob requires bumping it.
    #
    # Validity is mtime equality, not ordering, so a file restored from backup or
    # given an older timestamp invalidates too.
    #
    # ``key`` is the stored lookup key, which may be relative to the cache root.
    # ``path`` is only used to stat the file.
    return {
        "path": key if key is not None else str(path),
        "mtime": float(path.stat().st_mtime),
        "layout_version": LAYOUT_VERSION,
    }


def get_cached_layouts(paths: list[Path]) -> dict[Path, FileLayout]:
    if CACHE_DISABLED:
        return {}

    paths_by_cache_dir = sort_files_by_cache_dir(paths)
    layouts: dict[Path, FileLayout] = {}
    for cache_dir, cache_paths in paths_by_cache_dir.items():
        layouts |= read_layouts_from_cache(cache_dir, cache_paths)
    return layouts


def read_layouts_from_cache(
    cache_dir: Path, file_paths: list[Path]
) -> dict[Path, FileLayout]:
    if not file_paths:
        return {}

    # A file may be stored under an absolute or a cache-root-relative key, so
    # resolve every candidate key back to the path the caller asked for.
    paths_by_key: dict[str, Path] = {}
    for file_path in file_paths:
        for key in __candidate_cache_keys(cache_dir, file_path):
            paths_by_key.setdefault(key, file_path)

    try:
        entries = get_cache_entries(cache_dir, list(paths_by_key))
    except Exception:
        logger.debug("cache read failed", exc_info=True)
        return {}

    output: dict[Path, FileLayout] = {}
    for entry in entries:
        try:
            if entry.get("layout_version") != LAYOUT_VERSION:
                continue
            key = str(entry["path"])
            path = paths_by_key.get(key)
            if path is None or path in output:
                continue
            try:
                current_mtime = path.stat().st_mtime
            except OSError:
                continue
            if current_mtime != float(str(entry["mtime"])):
                continue

            layout = read_blob(cache_dir, key)
            if layout is None:
                continue
            # Output layout may be relative
            output[path] = replace(layout, path=path)
        except Exception:
            logger.debug("cache entry processing failed", exc_info=True)
            continue

    return output


def read_blob(cache_dir: Path, key: str) -> FileLayout | None:
    blob_uuid = get_string_uuid(key)
    blob_path = cache_dir / f"{blob_uuid}.json"
    if not blob_path.exists():
        return None

    try:
        from opencosmo.io import discover

        blob_bytes = blob_path.read_bytes()
        blob = json.loads(blob_bytes)
        return discover.decode_file_layout_blob(blob)
    except Exception:
        logger.debug("blob read failed", exc_info=True)
        return None


def get_cache_entries(cache_dir: Path, keys: list[str]) -> list[dict[str, object]]:
    try:
        placeholders = ",".join("?" for _ in keys)
        query = f"SELECT * FROM files WHERE path IN ({placeholders})"

        with open_cache_db_for_read(cache_dir) as conn:
            if conn is None:
                return []
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, keys)
            return [dict(row) for row in cursor.fetchall()]
    except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError):
        logger.debug("cache db read failed", exc_info=True)
        return []
