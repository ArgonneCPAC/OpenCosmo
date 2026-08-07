# AGENTS.md

This file provides guidance to coding agents working in this repository.

## Project Overview

OpenCosmo is a Python toolkit for reading, writing, and lazily transforming data from cosmological simulations produced by Argonne National Laboratory's CPAC group. It reads HDF5 files with full unit awareness and can materialize standard datasets as Astropy, NumPy, pandas, Polars, Arrow, or JAX containers. HEALPix maps use HealSparse or HEALPix array outputs.

## Repo Layout

The repository is a hybrid Rust/Python project built with **maturin**:

```text
src/                         — PyO3/Rust extension source
python/opencosmo/            — Python package
  dataset/                   — Dataset facade, immutable state, and materialization
  collection/                — Lightcone, map, structure, and simulation collections
  column/                    — Expressions, reducers, evaluation, and cache
  handler/                   — Raw data/cache protocols and HDF5 handlers
  index/                     — Index arithmetic and index-aware HDF5 reads
  io/                        — Discovery, file specs, MPI planning, and writers
  spatial/                   — Regions, trees, octree, and HEALPix indexing
  units/                     — Unit conventions and lazy conversion
  dtypes/                    — Pydantic header models and registries
  plugins/                   — Internal hook registry and contexts
  analysis/                  — CLI and analysis integrations
  serde/                     — Serialization and UI-facing models
test/                        — Serial and MPI tests
docs/source/                 — Sphinx documentation
changes/                     — Towncrier fragments
plans/                       — Design and implementation notes
```

The Rust extension compiles to `opencosmo._lib` and is imported by the Python `index/` module.

## Commands

### Setup
```bash
uv sync                          # Install all dev dependencies (compiles Rust extension)
uv sync --group mpi --group test-mpi  # Add MPI support
```

### Testing
Tests require the repository-local `test_data/` directory, which is distributed separately. Run:
```bash
uv run pytest --ignore=test/parallel          # All non-parallel tests
uv run pytest test/test_dataset.py            # Single test file
uv run pytest test/test_dataset.py::test_name # Single test
uv run mpiexec -n 4 pytest -m parallel test/parallel -x  # MPI tests
```

Test data fixtures are in `test/conftest.py`. The fixtures (`snapshot_path`, `lightcone_path`, `map_path`, `diffsky_path`, `analysis_path`) resolve to subdirectories under `test_data/`.

### Linting & Type Checking
```bash
uv run ruff check python/        # Lint
uv run ruff format python/       # Format
uv run mypy python/              # Type checking
```

### Changelog
```bash
uv run towncrier create <issue_number>.<type>  # Types: feature, bugfix, improvement, doc, removal, misc
```

## Coding Guidelines

- Prefer functional code. Make classes immutable whenever possible; user-facing classes must always be immutable. Prefer frozen dataclasses for backend containers.
- Avoid helper methods unless they are necessary. Prefix helper methods with `__` so they are name-mangled.
- Prefix backend module functions with `__` when they are not used outside their defining module.
- Avoid over-commenting. Add concise comments only when the logic is not obvious.
- Do not use class inheritance except where an existing framework requires it, such as Pydantic models.
- Add type hints to all code. Avoid `Any` whenever a more specific type can be used.
- Target Python 3.12+ syntax. Put annotation-only imports behind `TYPE_CHECKING` unless a framework such as Pydantic requires the object at runtime.
- Use `Protocol` for interchangeable backend or plugin contracts, `TypedDict` for established dictionary payloads, `NamedTuple` when tuple behavior matters, and frozen dataclasses for richer immutable records. Add `@runtime_checkable` only for actual runtime checks.
- Use narrow, rule-specific `# noqa` or `# type: ignore[...]` suppressions. Do not use a broad suppression when the type or lint rule can be expressed accurately.
- Add public names through the appropriate package `__init__.py` and `__all__`. Keep Rust bindings behind Python APIs rather than exposing `opencosmo._lib` directly.
- Keep optional dependencies out of import-time paths. Import pandas, Polars, Arrow, JAX, yt, and MPI implementations only when requested. Serial code must handle `get_comm_world() is None`.
- Discovery, schema construction, serialization, and MPI planning must be deterministic. Sort file, group, and column inputs and never depend on set iteration or caller path order where ranks must agree.
- Accept `str | Path` at public file boundaries, normalize to `Path`, and use context managers for owned HDF5 resources. Frozen discovery and planning records must not retain live h5py objects.
- Match existing exception semantics: `TypeError` for unsupported object kinds, `ValueError` for invalid domain values or combinations, filesystem-specific exceptions for paths, `UnitsError` for unit arithmetic, and `RuntimeError` for broken internal invariants. Treat changes to tested user-facing error messages as behavioral changes.
- Public APIs use NumPy-style docstrings compatible with Sphinx Napoleon. Update the relevant guide or reference page when user-visible behavior changes.

### Rust Guidelines

- Keep PyO3 functions as thin `_py` wrappers that validate Python inputs and delegate to typed Rust functions using `ArrayView1<i64>` or `Array1<i64>`. Preserve Python names with `#[pyfunction(name = "...")]` and return NumPy arrays through `IntoPyArray`.
- Preserve the repository-wide signed `int64` index invariant. Return Python exceptions for invalid input, handle empty arrays explicitly, and do not panic or use unchecked indexing for user-controlled shapes and ranges.

### Test Guidelines

- Prefer focused pure tests for planning and validation logic. Use shared fixtures for repository test data and `tmp_path` for generated files.
- Test transformation immutability and user-facing errors where relevant. Use `pytest.raises(..., match=...)` when the message is part of the contract.
- MPI tests belong under `test/parallel`, use `@pytest.mark.parallel(nprocs=4)`, and should use `parallel_assert` for rank-coordinated assertions.

## Architecture

### Core Data Model

`Dataset` (`python/opencosmo/dataset/dataset.py`) is the primary user-facing class. Transformations (`filter`, `select`, `take`, `bound`, `with_units`, etc.) return new logical datasets. Column materialization is generally lazy and producer-driven, but filters, sorting, exact spatial bounds, and structure-link operations may read the data they need before an explicit `get_data()` call.

`Dataset` wraps an immutable `DatasetState` (`dataset/state.py`) and an optional `Tree`. `DatasetState` is a frozen dataclass containing the producer graph, raw `DataHandler`, shared/derived `DataCache`, `UnitHandler`, header, visible column-to-producer UUID map, region, open kwargs, sort key, and metadata columns. Row selection lives on `raw_data_handler.index`. State transformations are standalone functions returning `dataclasses.replace(...)` copies; caches are intentionally mutable infrastructure.

### Handler Layer

`python/opencosmo/handler/` defines separate `DataHandler` and `DataCache` protocols. `DataHandler` provides indexed raw reads, row selection, load-condition metadata, and schema generation. `DataCache`, implemented by `ColumnCache`, stores UUID/name-keyed data and manages parent/child cache derivation and live-state registration. `Hdf5Handler` implements raw HDF5 access; `EmptyHandler` backs datasets whose values live entirely in a cache.

### Index Module

`python/opencosmo/index/` centralizes all index arithmetic. An index tracks which rows of a dataset are selected. Two forms exist:
- `SimpleIndex` — a 1-D `int64` numpy array of row positions
- `ChunkedIndex` — a `(start, size)` pair of 1-D `int64` arrays for contiguous-chunk reads

`DataIndex = SimpleIndex | ChunkedIndex`. Python modules provide type dispatch, validation, HDF5 reads, and most arithmetic. Performance-critical range, expansion, take, remapping, splitting, and projection kernels are implemented in `src/index.rs` and exposed internally through `opencosmo._lib.index`.

### Column Module

Column materialization uses UUID-identified producers. `RawColumn` represents disk or in-memory sources, `Column` represents algebraic expressions and masks, `EvaluatedColumn` wraps user functions, and `DerivedScalarValue` represents reductions. `DatasetState.column_map` maps visible names to producer UUIDs, allowing safe column replacement. `dataset/graph.py` builds the `rustworkx` dependency DAG, and `dataset/instantiate.py` fetches and evaluates only requested transitive dependencies.

`ColumnCache` persists values across related immutable states using parent/child caches and derived indices. `LocalReducer` handles in-process reductions. `MpiReducer` uses collective reductions for min/max/sum/mean and gather/broadcast for variance, standard deviation, median, and quantile.

### Collections

`opencosmo.open()` chooses a dataset or collection from the discovered HDF5 layout, not simply from the number of paths. A single multi-group file may produce a collection, and several redshift-step files commonly produce one `Lightcone`.

- `StructureCollection` models a properties source plus datasets linked through `/data_linked` start/size or index metadata. Collections may be nested, and linked datasets are rebuilt lazily after row transformations. Iteration uses `objects()`, with `halos()` and `galaxies()` aliases.
- `SimulationCollection` is a mapping of same-kind datasets or collections from independent simulation scopes. It maps operations across children. Redshift steps from one simulation belong in a `Lightcone`, not a `SimulationCollection`.
- `Lightcone` presents redshift-step datasets as one dataset-like object and may be nested by step and subtype. It stacks children at materialization time and supports redshift and HEALPix-pixel pruning.
- `LightconeScope` owns expressions that must run after children are stacked: scalar reductions and expressions depending on existing scope-owned names. Pure child expressions are routed to each child dataset.
- `HealpixMap` wraps map data plus explicit coverage metadata. It currently supports one layer and returns HealSparse objects or nested-order HEALPix arrays through map-specific resolution and cone-query paths.

Collection types are in `python/opencosmo/collection/`.

### Spatial Indexing

Spatial queries use `Dataset.bound(region)`. Regions are constructed via `make_box`, `make_cone`, or `make_skybox` (all public in `opencosmo`). The `spatial/` subpackage contains:
- `models.py` — Pydantic models for region parameters (`BoxRegionModel`, `ConeRegionModel`, etc.)
- `region.py` — concrete `Region` implementations (`FullSkyRegion`, `HealpixRegion`, etc.)
- `relations.py` — spatial predicates (`contains_2d`, `contains_3d`, `intersects_2d`, etc.)
- `builders.py` — `from_model()` factory; `protocols.py` — `Region` protocol
- `octree.py` — octree for 3D snapshot data (z-order curve, level-based start/size arrays)
- `healpix.py` — HEALPix index for 2D lightcone data (nested ordering)
- `tree.py` — `Tree` wrapper used by `Dataset`

`Tree.query()` separates fully contained chunks from boundary-intersecting chunks. `Dataset.bound()` accepts contained chunks directly and reads coordinates only for exact checks on boundary candidates. Ordinary snapshots and lightcones use octree or nested-HEALPix trees; `HealpixMap` uses explicit map coverage instead of this tree path.

### HDF5 File Format

OpenCosmo files are structured HDF5:
```
/data/           — columns (HDF5 datasets), each with optional "unit" and "description" attributes
/header/         — simulation metadata (cosmology, parameters)
/index/          — spatial index (level_0, level_1, ... each with start/size arrays)
/data_linked/    — links to associated datasets (e.g., haloparticles_start, haloparticles_size)
```
Each logical dataset group contains `/data` and may contain sibling `/index` and `/data_linked` groups. Headers may be at the root or nested in dataset/simulation scopes; discovery assigns each `/data` group to its nearest enclosing `/header`. Multi-dataset files therefore do not necessarily share one root header. See `SPEC.md` for the full format.

### Unit System

Units are handled through `UnitConvention` (`python/opencosmo/units/`). Four conventions exist: `scalefree`, `comoving`, `physical`, and `unitless`. `UnitHandler` tracks the file's base convention separately from the current convention and applies registered conversions when raw data is materialized. Convention changes invalidate affected cache entries. Header Pydantic models use the separate unit registry in `dtypes/units.py`; files open in the comoving convention by default.

### I/O Layer

Opening follows a discover/match/plan/build pipeline:

- `discover.py` walks each file once and creates frozen, picklable `FileLayout` and `GroupLayout` records without retaining HDF5 handles.
- `specs.py` contains the ordered core `FileSpec` registry for recognizing and validating structure collections, maps, lightcones, and datasets. File-type dispatch is intentionally not a plugin extension point.
- `plan.py` computes deterministic per-rank assignments, rehydrates live HDF5 targets, and delegates construction to the matched spec.
- `index_spec.py` defines row-index policies for spatial partitioning, full reads, and empty reference datasets.
- `iopen.py` orchestrates the pipeline and implements low-level `open_dataset()`.

Under MPI, spatial mode gives each rank the file set and partitions source rows. Redshift mode assigns contiguous, approximately volume-balanced redshift-step groups; empty ranks open an empty-schema reference. Unsupported nested layouts fall back to spatial distribution. Discovery performs the collective metadata exchange, after which every rank computes the same deterministic assignment.

Writing is schema-based. `schema.py` defines recursive `Schema` nodes, `writer.py` defines lazy column sources and combination strategies, and `serial.py`/`mpi.py` perform serial or collective HDF5 writes. `verify.py` and `updaters.py` support structural checks and metadata/index rewrites. Optional Parquet output is exposed as `opencosmo.io.write_parquet` when PyArrow is installed.

### Plugin Hooks

`python/opencosmo/plugins/` implements an in-process hook registry. `fold()` applies all matching transforms to frozen context dataclasses; `query()` returns the first matching non-`None` result. Hooks cover dataset/lightcone opening and instantiation, index updates, post-sort remapping, and MPI partition selection. Built-in lightcone and Diffsky hooks normalize columns and preserve host relationships. Core file recognition in `io/specs.py` remains separate from plugins.

### Top-level Utility Modules

Several concerns are now in dedicated top-level modules under `python/opencosmo/`:
- `header.py` — `OpenCosmoHeader`, `read_header()`
- `file.py` — path validation, HDF5 resource decorators, broadcast reads, and metadata utilities
- `mpi.py` — `get_comm_world()` (returns `None` when mpi4py is absent)
- `cosmology.py` — cosmology helpers
- `dtypes/` — Pydantic header models and origin/data-type registries
- `analysis/` — Click CLI plus Diffsky and yt integrations

### Key Patterns

- **Pydantic header models** live in `python/opencosmo/dtypes/`. Origin, data type, and lightcone status select the required and optional model groups exposed by `OpenCosmoHeader`.
- **`from __future__ import annotations`** is used where needed. `TYPE_CHECKING` guards keep annotation-only imports out of runtime paths.
- **Missing third-party stubs** are configured in `pyproject.toml`; source-level suppressions should remain narrow and local.
- **`rustworkx`** is used for dependency graphs (e.g., derived column evaluation order).
- The CLI entrypoint is `opencosmo.analysis.cli:cli` (accessible as `uv run opencosmo`).
- Version changes must keep `pyproject.toml`, `Cargo.toml`, `opencosmo.__version__`, Sphinx `release`, and `.bumpversion.toml` synchronized.
