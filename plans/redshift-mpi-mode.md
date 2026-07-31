# Redshift-based MPI mode for lightcones

## Context

When OpenCosmo opens data under MPI today, it **always splits spatially**: every
rank opens a handle to *every* file and reads a spatial sub-partition of each
(`open_single_dataset` → `dataset.mpi.partition` → `Tree.partition`, in
`io/iopen.py:555` and `dataset/mpi.py:19`). For lightcones this is pathological.
A lightcone is a set of files, one per redshift step, and there can be 100+ of
them. With 32 ranks that is 32 × 100 file opens — an enormous amount of
filesystem metadata traffic, most of it redundant.

For lightcones a far better decomposition is **by redshift step**: distribute the
files across ranks so each rank reads roughly equal data volume, and each file is
opened by exactly one rank. The write path already tolerates ranks holding
*different* children/columns (via unioned keys and sub-communicators in
`io/mpi.py`), so the written output is a single, correct, combined lightcone.

This plan adds a user-selectable `mpi_mode` argument to `open()` that switches a
lightcone open from spatial partitioning to redshift-step distribution.

### Decisions
- **API:** explicit `mpi_mode` keyword on `open()`. Enum with `"spatial"`
  (default, current behavior) and `"redshift"`.
- **Balance metric:** total **row count** per file (from `column.shape[0]`).
  Cheap, needs no extra filesystem stat calls, directly reflects object count.
  *(Open to changing to bytes-on-disk or row×width if you prefer — called out here.)*
- **Scope (first pass):** plain lightcones (one properties dataset per step) **and**
  lightcone structure-collections (linked particles/profiles travel with their
  parent step to the same rank). **Nested Diffsky (step → type) lightcones are not
  redshift-split in this pass: when rank 0 detects a nested lightcone during
  planning, the open falls back to the existing spatial planner** (every rank opens
  every file and partitions spatially, as today). Redshift-splitting nested
  lightcones is a flagged follow-up.
- `mpi_mode="redshift"` is only valid for lightcones; anything else raises. With
  no MPI communicator (`get_comm_world() is None`) the argument is a no-op and
  files open normally on the single process.
- **Empty ranks still open one reference file, with an empty index.** A rank that
  the planner assigns no data does *not* build a truly empty lightcone. Instead it
  opens a single designated reference file and holds a zero-length index
  (`index.build.empty()`). This keeps the full schema (column names, dtypes,
  units, `z_range`) available on every rank, so local ops like `select` / `filter`
  and MPI reductions behave identically everywhere. (Required anyway:
  `Lightcone.__init__` calls `next(iter(self.values()))` at `lightcone.py:110-112`
  and cannot be constructed from an empty dict.)

## The one real correctness gap

`Lightcone.make_schema` (`collection/lightcone/lightcone.py:591`) loops over
**this rank's local steps** and calls `stack_lightcone_datasets_in_schema`
(`collection/lightcone/stack.py:148`) per step. That callee contains MPI
collectives (`get_all_keys(..., comm)` and `sync_headers`). Today every rank has
every step, so the call counts match. Under a redshift split, rank A holds steps
{600,601} and rank B holds {700,701} — the loop runs a different number of times
per rank, the collectives fall out of lockstep, and the write **deadlocks**.

Fix: drive the top-level step loop from the **cross-rank union** of step keys
(same pattern already used inside `stack.py` for ds-group keys and for children in
`io/mpi.py`), taking the existing empty-step branch for steps a rank doesn't hold.

## Tasks

### 1. Add `mpi_mode` argument + enum
- `io/io.py`: add `mpi_mode: str = "spatial"` to `open()` signature and docstring.
  Define an enum (e.g. `MpiMode(Enum)` in `io/io.py` or a small `io/modes.py`)
  with `SPATIAL` / `REDSHIFT`; accept the string and normalize. Pass it into
  `open_files`.
- `io/iopen.py`: add `mpi_mode` parameter to `open_files` (default spatial).
  Only the multi-file branch (`open_files:89-91`) consults it.

### 2. Distribution planner + compatibility verification (new module `collection/lightcone/distribute.py`)
- `plan_redshift_distribution(paths, comm) -> DistributionPlan`, computed on rank 0
  and broadcast. The plan carries, per rank, the list of assigned paths **and** a
  single `reference_path` (used by empty ranks — see below).
- **Rank 0 is now the sole place all files are seen, so it must do the
  compatibility verification that every-rank-opens-every-file gave us for free
  today:**
  - Open each path once, read `read_header(f)` for `file.step` /
    `file.is_lightcone` / `file.data_type`, the ordered **column set + dtypes**
    from `/data` (and each linked `/<group>/data` for structure-collections), and
    the row count from the first `/data` column's `.shape[0]`.
  - Assert every file `is_lightcone`; assert all files share the **same column
    names, dtypes, and data_type** (raise a clear error naming the offending file
    otherwise). This is the check that previously happened implicitly on all ranks
    and in `Lightcone.__init__` (`lightcone.py:107-111`).
  - Detect nested Diffsky (multiple types per step). If found, **signal fallback**:
    the planner returns a sentinel (e.g. `None`) rather than a distribution plan,
    and `open_files` reverts to the normal spatial path
    (`__determine_multi_file_collection_type` → collection `.open`, every rank
    opening every file). This decision is made on rank 0 and broadcast so all ranks
    take the same branch.
- Greedy longest-processing-time bin-pack: sort files by descending row count,
  assign each to the currently-least-loaded rank. Keep files for the same step
  together; for structure-collection lightcones, keep a step's linked datasets
  with their parent.
- Pick a `reference_path` (e.g. the smallest file) and include it in the plan for
  every rank. `comm.bcast` the full plan.
- Reuse `read_header` from `header.py`; do not read column data (shapes/dtypes
  only).

### 3. Redshift-open branch in `open_files`
- When `mpi_mode == REDSHIFT`, `comm is not None`, and `len(paths) > 1`:
  - Call the planner. **If it signals fallback (nested lightcone detected), take
    the normal spatial path** — `__determine_multi_file_collection_type(...).open(...)`
    exactly as `mpi_mode="spatial"` would — and skip the rest of this branch.
  - Otherwise select `my_paths = plan.paths[rank]`.
  - **Non-empty rank:** build `FileTarget`s via `__make_file_target` for
    `my_paths` only (this is the win: one rank per file), then dispatch to
    `Lightcone.open(valid_targets, redshift_split=True, ...)`.
  - **Empty rank (`my_paths == []`):** build a `FileTarget` for the single
    `plan.reference_path` and dispatch to
    `Lightcone.open([ref_target], redshift_split=True, empty=True, ...)`. The
    rank thus holds a full-schema lightcone with a zero-length index — every
    column is known, so `select`/`filter`/reductions work identically to a
    non-empty rank.
  - Do **not** run `__determine_multi_file_collection_type` per-rank (empty ranks
    have only the reference file); the planner already asserted lightcone, so
    force that path.

### 4. Whole-file read + empty-index for redshift-split datasets
- `Lightcone.open` (`lightcone.py:469`): accept `redshift_split: bool = False`
  and `empty: bool = False`.
  - When `redshift_split` and not `empty`: call
    `iopen.open_single_dataset(..., bypass_mpi=True)` so each file is read whole
    on its owning rank (the `bypass_mpi` path at `iopen.py:555` already exists and
    leaves `index=None` → full read).
  - When `empty`: open the reference dataset but force a zero-length index. Add a
    thin path so the resulting `DatasetState` uses `index.build.empty()`
    (`empty` is already imported at `iopen.py:18` and used at `iopen.py:560`)
    instead of a full read. The dataset keeps all columns/units/`z_range` but 0
    rows.

### 5. Synchronize `Lightcone.make_schema` step loop (critical write fix)
- `lightcone.py:603`: replace `for step, datasets in output_datasets.items()`
  with iteration over `get_all_keys(output_datasets, get_comm_world())` (import
  from `io/mpi.py`). For each step in the union, use
  `output_datasets.get(step, {})`; empty → take the existing empty-step branch
  (`lightcone.py:604-606`), non-empty → the current zrange + stack path.
  This guarantees every rank calls `stack_lightcone_datasets_in_schema` the same
  number of times in the same order.
- Confirm `sync_headers`' empty-participation branch (`stack.py:113-117`) already
  covers steps a rank lacks (it does) — no change expected there.

### 6. Empty-rank lightcone (reference file + empty index)
- No sentinel/None `z_range` hack is needed: an empty rank opens the reference
  file, so `Lightcone.__init__` (`lightcone.py:90-113`) runs normally —
  `next(iter(self.values()))` (`lightcone.py:110-112`) has a real dataset, and the
  column-equality check (`lightcone.py:107-111`) passes. The reference file's real
  `z_range` is later reconciled to the global range across ranks at write time in
  `sync_headers` (`stack.py:135-138`).
- Verify the zero-length dataset flows correctly through
  `LightconeScope`/`MpiReducer`: an empty rank must still contribute a valid
  length-0 local reduction (reduce-locally-then-combine invariant), not skip the
  collective. Add a test (task 7) rather than assuming.

### 7. Tests (`test/parallel/test_lc_mpi.py`)
- `@pytest.mark.parallel(nprocs=4)`:
  - Open a multi-step lightcone with `mpi_mode="redshift"`; assert each rank holds
    a **disjoint** subset of steps and their union equals the full step set.
  - Write round-trip: write the distributed lightcone, reopen serially, assert row
    counts and z_range match a serial open of the same files.
  - **Files fewer than ranks** (e.g. 2 files, 4 ranks): assert the empty ranks
    hold a reference dataset with **len 0 but full columns**, that `select`/
    `filter` succeed on those ranks, and that the write still succeeds (exercises
    the zero-length-rank exclusion at `io/mpi.py:85-94` + the make_schema union fix).
  - A scalar reduction across the lightcone (e.g. `col("mass").mean()`) returns
    the same value in redshift mode as spatial/serial, **including** the
    files-fewer-than-ranks case (validates empty ranks contribute a valid length-0
    reduction and `LightconeScope` reductions still span all ranks).
- Guard: `mpi_mode="redshift"` on a non-lightcone raises a clear error.
- Guard: incompatible files (mismatched columns/dtypes) raise on rank 0 during
  planning, naming the offending file.

### 8. Changelog + docs
- `uv run towncrier create <issue>.feature` describing the new `mpi_mode`.
- Document `mpi_mode` in the `open()` docstring and any lightcone/MPI user docs.

## Verification
- `uv run ruff format` (touched files only) and `uv run ruff check python/`;
  `uv run mypy python/` package-wide.
- Serial sanity: `uv run pytest test/test_lightcone*.py` (ensure default
  spatial path unchanged).
- Parallel: `uv run mpiexec -n 4 pytest -m parallel test/parallel/test_lc_mpi.py -x`
  and again with `-n 2` and a >2-step lightcone to exercise real distribution.
- Manual smoke: open a many-file lightcone under `-n 4` with `mpi_mode="redshift"`
  and confirm (e.g. via logging/strace or open-count) each file is opened by one
  rank, not all four.

## Risks / follow-ups
- **Nested Diffsky (step → type) lightcones** are out of scope this pass. The
  planner detects them on rank 0 and falls back to the spatial planner (every rank
  opens every file), so `mpi_mode="redshift"` is safe to pass — it silently
  degrades rather than mis-distributing. Redshift-splitting nested lightcones is a
  follow-up. Add a test asserting the fallback opens correctly and produces the
  same result as spatial mode.
- **Balance quality** with row-count-only metric may be off if datasets have very
  different column widths; revisit metric if imbalance is observed.
- **Linked-dataset co-location** for structure-collection lightcones must be
  enforced in the planner (a step's particles/profiles cannot land on a different
  rank than its properties).
