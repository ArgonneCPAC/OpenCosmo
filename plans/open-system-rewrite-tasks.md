# Phase 1 — Atomic task sequence (spec-based, MPI-unified file opening)

Derived from `plans/open-system-rewrite.md`. Scope is **Phase 1 only**; the kwarg
cleanup in `plans/open-system-rewrite-followup.md` is a separate later PR.

Each task below is a **self-contained, independently reviewable PR-sized unit** meant to be
handed to an implementation agent. Tasks are ordered by dependency. The build interfaces
(`open_single_dataset`, `StructureCollection.open`, `Lightcone.open`,
`build_structure_collection`, and their `redshift_split`/`empty`/`index_override`/`bypass_*`/
`metadata_group` kwargs) **stay stable** through all of Phase 1 — do not change their signatures.

## Global guardrails (apply to every task)

- **Do not touch the Rust extension** (`src/`). Metadata-only discovery reads header fields, group
  paths, column names/dtypes, and row counts — all available from h5py.
- **Do not change any collection build-interface signature.** Phase 1 feeds the *existing* builders
  through a new front-end. Signature cleanup is the follow-up PR.
- **Determinism is a correctness requirement.** Every rank must compute identical results from
  identical inputs. Sort layouts by path; use `sorted()`/`frozenset` in all match/verify/distribute
  logic (mirrors the existing "sorted for lockstep" note at `structure/io.py:404-409`).
- **Collectives must stay in lockstep.** Discovery captures structural failures into
  `FileLayout.error` instead of raising mid-collective. Planning is broadcast-free by determinism.
- Per-task verification: `uv run ruff format` (touched files only), `uv run ruff check python/`,
  `uv run mypy python/` (package-wide). Add tests as specified.

## Dependency graph

```
T1 discover.py ──> T2 plan.py ──> T3 specs.py ──> T4 flip open_files ──> T5 SPEC.md + changelog
```

T2 depends on T1. T3 depends on T1+T2. T4 depends on T1–T3. T5 depends on T4.

---

## T1 — Stage 1: metadata-only discovery (`io/discover.py`)

**Goal.** Walk each file once and produce a frozen, picklable layout with no live h5py handles.
Purely additive — does not touch `open_files` yet.

**New file:** `python/opencosmo/io/discover.py`

**Implement:**
- `@dataclass(frozen=True) GroupLayout` with the fields from the plan: `path: str`,
  `header: OpenCosmoHeader`, `column_names: tuple[str, ...]`, `column_dtypes: tuple[str, ...]`,
  `row_count: int`, `has_index: bool`, `linked_target_names: tuple[str, ...]`. (`load/if` is
  re-evaluated live at build time — not serialized.)
  - **Add `header_path: str`** — the in-file path of the group's *governing* header (the deepest
    `/header` whose group is an ancestor of this `/data` group; the "nearest enclosing header" rule
    already used by `__find_datasets_under_group` / `__get_collection_dataset_groups`). This is the
    grouping key for reconstructing composition (see "Nesting" below); it must NOT be inferred from
    header *content*. For a single-header file it is `/header`; for a nested sim collection it is
    `/scidac1/header`, `/scidac2/header`, etc.
- `@dataclass(frozen=True) FileLayout` with `path: Path`, `groups: tuple[GroupLayout, ...]`,
  `error: Optional[str] = None`.
- `discover_file(path) -> FileLayout`: reuse the existing walk (`__make_group_map` from
  `iopen.py:124`) and read the header **locally** via `read_header.__wrapped__` (the undecorated
  read — no world-comm bcast). Fill columns/dtypes/row_count by porting the metadata helpers
  currently in `collection/lightcone/distribute.py`: `__get_columns_info` (distribute.py:347) and
  `__get_data_row_count` (distribute.py:374). Capture any structural failure into
  `FileLayout.error` (empty string / malformed / missing header) — **do not raise**.
- `discover_all(paths, comm) -> tuple[FileLayout, ...]`: distributed round-robin — rank `i`
  discovers `paths[i::nranks]`, then a single `comm.allgather` assembles the full list on every
  rank; **sort the assembled tuple by path** so every rank holds an identical ordering. Serial or
  single-file → only rank 0 walks (no allgather).
- Factor the reusable **inference helpers** off the current `iopen` privates into pure functions
  that operate on `GroupLayout`/`FileLayout` (not h5py). These are the predicates the specs will
  reuse in T3 — e.g. "is this group a lightcone group", "does this group carry `/data_linked`",
  "data_type of this group". Source logic: `__identify_group_types` (iopen.py:406),
  `__determine_multi_file_collection_type` (iopen.py:278). Keep them small and side-effect-free.

**Nesting (how the flat tuple represents a `SimulationCollection` of `StructureCollection`s).**
The `groups` tuple holds the *leaf* `/data` nodes only; composition is reconstructed, not stored as
a tree. A nested sim collection is two layers deep in one file (`/scidac1/halo_properties/data`,
`/scidac1/halo_particles/data`, `/scidac2/halo_properties/data`, …) with one header per simulation
(`/scidac1/header`, `/scidac2/header`). Reconstruction is **recursive** and keyed on
`(file.path, group.header_path)`:
1. bucket `GroupLayout`s by governing `header_path` → one bucket per simulation scope;
2. >1 distinct top-level header scope → `SimulationCollection`, recursing into each scope;
3. within a scope, the ordinary composition rules apply (properties + `/data_linked` + links →
   structure collection; single group → dataset; etc.).
This exactly reproduces today's `__get_collection_dataset_groups` (nearest-enclosing-header
attach) + `__combine_dataset_groups` (bucket by top-level name) +
`__open_dataset_targets_for_sim_collection` (build one collection per bucket, wrap in
`SimulationCollection`). Note `SimulationCollection.open` raises `NotImplementedError` — a sim
collection is only ever *assembled* from already-built children, never opened via a builder. The
inference helpers factored out here must operate on `header_path` + group paths, never on header
content equality (two simulations can share simulation parameters).

**Do not:** modify `open_files` or delete any `iopen` private in this task. The old path keeps
working; this file is dormant until T4.

**Tests (new):**
- `test/test_discover.py`: single-file layout for each fixture (`snapshot_path`, `lightcone_path`,
  `map_path`, `diffsky_path`), plus a serial multi-file discover; assert column names/dtypes/
  row_count/has_index/linked_target_names match the file, and that a deliberately malformed file
  yields `error` set (not an exception).
- `test/parallel/test_discover_mpi.py` (marked `parallel`): assert every rank returns byte-identical
  path-sorted layouts; assert round-robin coverage; assert the fewer-files-than-ranks case works.

**Acceptance:** new tests pass serial (`uv run pytest test/test_discover.py`) and MPI
(`uv run mpiexec -n 4 pytest -m parallel test/parallel/test_discover_mpi.py`, plus `-n 2`); existing
suite unaffected; lint/type clean.

---

## T2 — Stage 3 mechanism: `Assignment`, distribute, build-from-assignment (`io/plan.py`)

**Goal.** The deterministic, broadcast-free distribution + rehydration mechanism the specs will
drive. No spec policy here — this is the generic machinery.

**New file:** `python/opencosmo/io/plan.py`. **Depends on:** T1.

**Implement:**
- `@dataclass(frozen=True) Assignment` per the plan: `rank: int`,
  `file_indices: tuple[int, ...]` (into the sorted layouts tuple), `index_kind:
  Literal["spatial", "redshift_step", "none"]`, `is_empty_ref: bool = False`.
- A `distribute(layouts, mpi_mode, nranks) -> tuple[Assignment, ...]` helper family:
  - `spatial`: every rank gets every file; `index_kind="spatial"`.
  - `redshift`: reuse `partition_contiguous` (keep it in `distribute.py` as-is) on per-step row
    counts read from the layouts → contiguous step chunks; empty ranks get the lightest step with
    `is_empty_ref=True`. Nested diffsky-style lightcone → fall back to `spatial` (today's `None`
    sentinel path). Port the step-grouping/weight logic from
    `__compute_redshift_distribution_plan` (distribute.py:172) but operating on **in-memory
    layouts**, not re-reading files.
  - serial / single file → rank 0 gets all.
  - Every rank computes the identical `Assignment` tuple from the identical sorted layouts — **no
    `comm.bcast`**. Malformed input → every rank raises the same `ValueError` by construction.
- `build_from_assignment(assignment, layouts, matched_spec, open_kwargs) -> Dataset | Collection`:
  reopen **only this rank's** files, rehydrate the existing `FileTarget`/`DatasetTarget` TypedDicts
  (iopen.py:53-75) from the live handle + layout by navigating to the known `/data` paths (no full
  re-walk, no header re-read — reuse the already-discovered `OpenCosmoHeader`), re-run
  `evaluate_load_conditions` (iopen.py:716) live, then delegate to the matched spec's builder with
  today's kwargs derived from the `Assignment`:
  - `redshift_step` → `redshift_split=True`, `empty=assignment.is_empty_ref`.
  - `spatial`/`none` → today's defaults.

**Reference facts (verified):**
- `partition(comm, header, index_group, data_group, tree, min_level=None) -> Optional[TreePartition]`
  (`dataset/mpi.py:19`); `TreePartition` has `.idx`, `.region`, `.level`. Spatial resolution stays
  in the **build** stage (needs live h5py + tree) — `plan.py` only records `index_kind`.
- Empty index is `opencosmo.index.build.empty()` (returns a `ChunkedIndex`); `from_range(start,
  end)` and `single_chunk(start, size)` also live there. There is no `make_empty_index` — it is an
  import alias (`from ... import empty as make_empty_index`) in `structure.py`.

**Do not:** change builder signatures; do not add a broadcast step; do not resolve the spatial
partition here (that happens live in the builders during T4's flip).

**Tests (new):** `test/test_plan.py` — feed mock `FileLayout` tuples, assert:
- spatial assigns all files to all ranks;
- redshift produces contiguous, coverage-complete step chunks (can reuse the invariants in
  `test/test_distribute.py`); empty ranks flagged `is_empty_ref`;
- assignments are a pure function of sorted layouts (shuffling input order yields identical output).

**Acceptance:** `test/test_plan.py` and existing `test/test_distribute.py` pass; lint/type clean.

---

## T3 — Stage 2: spec registry (match + verify + distribute/build wiring) (`io/specs.py`)

**Goal.** One self-describing unit per file/collection type, in a **self-contained registry** that
lives entirely in the io layer.

**Design decision (do not use the plugin hook system).** Opening files is **core logic**, not an
extension point. The plugin machinery (`plugins/hook.py`, `HookPoint`) is for third-party/per-data-type
logic that *modifies* core behavior — it must not become the internal dispatch table for the open
pipeline. So T3 does **not** add a `HookPoint`, a `FileSpecCtx`, or register anything on the shared
`_registry`. Instead the registry is a module-level ordered tuple in `io/specs.py`, and dispatch is a
plain `match_spec(layouts)` function. This reproduces `query()`'s exact "first registration whose
predicate is true wins" semantics (`plugins/hook.py:60-74`) with a two-line `next(...)` over an
explicit list — and the whole precedence order is auditable in one place. A side benefit: no
"import for side effect" site is needed (the literal registry is populated the moment the module is
imported), so `plugins/` is untouched.

**New file:** `python/opencosmo/io/specs.py`. **Depends on:** T1, T2. **No edits to `plugins/`.**

**Design note (as built — no base-class inheritance).** Specs are **stateless** and describe only
what varies per type. They do **not** carry `distribute`/`build`: those were pure forwards to the
`plan.py` free functions, so there is no `BaseFileSpec` and no inheritance (matches the user's
convention — Protocol + standalone classes + free functions, never a base class to dedupe forwards).
The orchestrator (T4 `open_files`) calls `plan.distribute` / `plan.build_from_assignment` directly.
Likewise a **simulation collection is not a spec**: multi-scope decomposition lives in the free
function `group_by_scope`, so there are only **four** specs and `match_spec` only ever dispatches on
a single-scope layout (no recursion, no recursion-guard).

**Implement:**
- Module-top imports (safe, no cycle): the inference helpers + `FileLayout` from
  `opencosmo.io.discover` (`GroupLayout` under `TYPE_CHECKING` — only in annotations). **No import of
  `plan.py`** — specs don't touch it. Builder calls (`open_single_dataset`,
  `StructureCollection.open`, `Lightcone.open`) go **inside method bodies** as lazy imports (mirrors
  how `iopen.py` uses `occ`/`sc` only in bodies).
- `FileSpec` protocol with `name`, `matches(layouts) -> bool`, `verify(layouts) -> None` (raises
  `ValueError`), and `build_from_targets(...)` (must satisfy `plan.SpecBuilder` structurally — do not
  redefine it). No `distribute`/`build` on the protocol.
- `group_by_scope(layouts) -> dict[str, tuple[FileLayout, ...]]`: a free function that buckets groups
  by governing `header_path` scope (see T1 "Nesting"), returning an ordered `scope_name -> sub-layouts`
  mapping (one entry `"/"` for a single-scope open; one per simulation for a nested file). Each
  sub-layout is a `FileLayout` with `groups` filtered to that scope. Errored files are skipped
  (callers raise on `error` before grouping). This is what T4's orchestrator loops over.
- `_build_single_dataset(targets, open_kwargs)`: a **module-level** helper (not a method) shared by
  `DatasetSpec` and `HealpixMapSpec` — `open_single_dataset(targets[0]["dataset_targets"][0], ...)`.
- The registry and dispatch:
  ```python
  SPECS: tuple[FileSpec, ...] = (
      StructureCollectionSpec(), HealpixMapSpec(), LightconeSpec(), DatasetSpec(),
  )
  def match_spec(layouts): return next((s for s in SPECS if s.matches(layouts)), None)
  ```
  Order is most-constrained-first and **is** the precedence (preserves today's decision order in
  `__open_single_file` iopen.py:194-253 and `__get_collection_type_from_categorized_lists`
  iopen.py:326).
- The four concrete specs (each a standalone class conforming to `FileSpec`; `matches` operates on a
  single-scope layout):
  | Spec | Match signal | Builds via |
  |---|---|---|
  | `DatasetSpec` | 1 group, not lightcone, not healpix | `open_single_dataset` |
  | `HealpixMapSpec` | 1 group, `data_type == "healpix_map"` | `open_single_dataset` → `__open_healpix_map` |
  | `LightconeSpec` | ≥1 group, single data_type, all `is_lightcone` | `Lightcone.open` |
  | `StructureCollectionSpec` | properties-with-`/data_linked` group **and** >1 distinct data_type present | `StructureCollection.open` |
  - `HealpixMapSpec.build_from_targets` is identical to `DatasetSpec`'s — `open_single_dataset`
    already routes `healpix_map` to `__open_healpix_map` (`iopen.py:668-669`). It is a distinct spec
    purely for the match signal + documentation; both share the `_build_single_dataset` helper.
  - **`StructureCollectionSpec` requires >1 distinct data_type present**, not merely a
    properties-with-`/data_linked` group. A lone properties file (or several properties files of one
    type across redshift steps) carries `/data_linked` referencing children that are *not* in the
    open set — old `oc.open` opens it as a `Dataset`/`Lightcone`, not a structure collection. The
    collection only exists once ≥1 linked child type is actually opened alongside (verified against
    the real fixtures).
- `matches` uses only the T1 inference helpers (`group_data_type`, `is_lightcone_group`,
  `has_linked_targets`, `is_properties_group`) — never header-content equality, never live h5py.
  `StructureCollectionSpec` (not `LightconeSpec`) catches the lightcone-structure-collection because
  `LightconeSpec.matches` requires a *single* data_type while a structure collection has several
  (preserves the ordering note at `iopen.py:224-232`).
- **Port cross-file verification** from `distribute.py` into `verify()`, running on the in-memory
  gathered layouts (no extra I/O). Factor a shared helper
  `_verify_columns_consistent_per_datatype(layouts)` (matching column names/dtypes across steps for a
  given data_type — from `__compute_redshift_distribution_plan` distribute.py:217-236), called by
  `LightconeSpec.verify` and `StructureCollectionSpec.verify`. `StructureCollectionSpec.verify` also
  ports the structure-collection consistency **raises** from `__determine_collection_kind`
  (distribute.py:289-344): identical data_type sets across steps, every step has a
  properties-with-`/data_linked` file. (The *classification* half — `is_plain`/`any_properties_link`
  — already lives in `plan._distribute_redshift`.) Verification depth is **structural + cross-file
  only** — no link-index bounds / row-count / coverage checks.

**Do not:** touch `plugins/` (no `HookPoint`, no `FileSpecCtx`); wire specs into `open_files` yet
(that is T4); delete any old `iopen` private; or add a side-effect import site. Specs are exercisable
via a direct `from opencosmo.io.specs import match_spec` in a unit test before the flip.

**Tests (new):** `test/test_specs.py` — build mock `FileLayout` tuples for each single-scope type and
assert `match_spec` returns the correct spec; assert precedence (a structure-collection layout matches
`StructureCollectionSpec`, not `LightconeSpec`; a lone/multi properties-with-link layout matches
`LightconeSpec`/`DatasetSpec`, not `StructureCollectionSpec`); assert `group_by_scope` splits a nested
multi-scope layout into per-simulation single-scope sub-layouts (each matching a leaf spec) and skips
errored files; assert `verify()` raises on cross-step column/dtype mismatch and on inconsistent
structure-collection type sets.

**Acceptance:** `test/test_specs.py` passes; `match_spec` returns the right spec for each mock;
`plugins/` unchanged; lint/type clean; existing suite still green (old path untouched). *(Done: 15
tests pass; spot-checked against real fixtures via `discover_all → group_by_scope → match_spec`.)*

---

## T4 — Flip `open_files` to the new pipeline; delete the old dispatch (`io/iopen.py`)

**Goal.** Route the real entry point through discover → group by scope → (per scope) match spec →
verify → distribute → build → wrap, and delete the superseded machinery. This is the behavioral
integration; the regression suite is the gate.

**Files:** `python/opencosmo/io/iopen.py`, `collection/lightcone/distribute.py`; add a changelog
fragment. **Depends on:** T1–T3.

**Note on the T3 shape (as built).** Specs are stateless and describe only what varies per type
(`name`/`matches`/`verify`/`build_from_targets`); they do **not** carry `distribute`/`build`. The
orchestrator calls the `plan.py` free functions `plan.distribute` and `plan.build_from_assignment`
directly, passing the matched spec as the builder. A simulation collection is **not** a spec: a
layout with more than one governing header scope is decomposed by `group_by_scope` (in `io/specs.py`)
into one single-scope sub-layout per simulation, each built through the ordinary path, and the
results are wrapped in a `SimulationCollection`. So `match_spec` only ever sees a single-scope layout,
and there is no recursion or recursion-guard.

**Implement:**
- Rewrite `open_files` (iopen.py:78) body to:
  1. `layouts = discover_all(paths, comm)`; if any `layout.error` is set, raise a combined
     `ValueError` (every rank sees the same errors → collective).
  2. `scopes = group_by_scope(layouts)` (from `io/specs.py`); if empty, raise ("No valid
     datasets found!" parity).
  3. For each `scope_name` in `sorted(scopes)` (lockstep-deterministic), with `sub = scopes[scope_name]`:
     - `spec = match_spec(sub)`; raise if `None`.
     - `spec.verify(sub)`.
     - `assignments = plan.distribute(sub, mpi_mode, nranks)`; pick this rank's `Assignment`.
     - `children[scope_name] = plan.build_from_assignment(assignment, sub, spec, open_kwargs)`.
  4. `return next(iter(children.values()))` if there is one scope, else
     `SimulationCollection(children)` via the **direct constructor** (`SimulationCollection.open`
     raises `NotImplementedError`, `simulation.py:65-66`). This replaces `__combine_dataset_groups` +
     `__open_dataset_targets_for_sim_collection`.
- **Delete** the superseded privates and the redshift early-return:
  `__make_file_target` (iopen.py:134), `__find_all_datasets` (440), `__identify_group_types` (406),
  `__determine_multi_file_collection_type` (278), `__get_collection_type_from_categorized_lists`
  (326), `__identify_lightcone_type` (359), `__get_multi_dataset_type` (381), `__open_single_file`
  (194), `__open_dataset_targets_for_sim_collection` (256), `__open_files_redshift_split` (164), and
  the `mpi_mode.value == "redshift"` early-return block (iopen.py:92-108). Move any still-needed
  helpers (`__make_group_map`, `__find_datasets_under_group`, `__combine_dataset_groups`,
  `__get_collection_dataset_groups`, `__find_all_headers`) into `discover.py`/`plan.py` as used, or
  keep them if `build_from_assignment` rehydration reuses them.
- **Keep** `open_single_dataset` (586), `__open_healpix_map` (678), `__expand_lightcone_region`
  (705), `evaluate_load_conditions` (716) — unchanged signatures.
- In `distribute.py`: remove the `read_header.__wrapped__` desync hack and the rank-0-compute +
  `comm.bcast(plan)` dance (`plan_redshift_distribution` distribute.py:112-169,
  `__compute_redshift_distribution_plan` distribute.py:172). **Keep `partition_contiguous`** (it is
  the reusable linear-partition primitive T2 calls). Delete `DistributionPlan` and the metadata
  helpers now living in `discover.py`. The `broadcast=False` / `read_header.__wrapped__` code in
  `__find_all_datasets` also disappears with the privates.
- Remove the now-dead `broadcast` parameter plumbing and the redshift branch in `io/io.py` if it
  only existed to reach the old early-return (confirm `MpiMode` handling still selects redshift vs
  spatial — that selection now feeds `plan.distribute`).

**Do not:** change any builder signature; do not alter `open_single_dataset`'s body semantics
(Phase 1 keeps `bypass_*`/`index_override`/`redshift_split`/`empty` alive — they are removed in the
follow-up PR).

**Verification (this is the gate):**
- Serial regression, must pass unchanged: `uv run pytest --ignore=test/parallel` — especially
  `test_dataset.py`, `test_collection.py`, `test_structure_collection.py`, `test_lightcone.py`,
  `test_healpixmap.py`, `test_diffsky.py`, `test_distribute.py`, `test_header.py`, `test_file.py`.
- MPI regression, both strategies: `uv run mpiexec -n 4 pytest -m parallel test/parallel -x` —
  especially `test_lc_mpi.py`, `test_structure_collection_mpi.py`, `test_dataset_mpi.py`,
  `test_healpixmap_mpi.py`, `test_lc_scope_mpi.py`. Also run `-n 2` and `-n 8` to exercise
  empty-rank and imbalance paths.
- `uv run towncrier create <issue>.improvement` with a fragment describing the unified pipeline.
- Lint/type clean.

**Acceptance:** full serial + MPI suites green at `-n 2/4/8`; the old privates are gone; only one
collective (`allgather` in discovery) remains in the open path.

---

## T5 — Rewrite `SPEC.md`

**Goal.** Document each file/collection type and its identification signals (currently only a single
generic dataset is documented).

**Files:** `SPEC.md`. **Depends on:** T4 (so the documented signals match the shipped specs).

**Implement:** For each type — dataset, healpix map, lightcone, structure collection, simulation
collection (incl. nested SC) — document the HDF5 layout (`/data`, `/index`, `/data_linked`,
`/header`, `step`, healpix/lightcone header blocks, nested simulation-collection group layout) and
the identification signals: the inference rules (`data_type`, `is_lightcone`, presence of
`/data_linked`, multiple headers in distinct top-level groups). Cross-reference the four spec names
from `io/specs.py` (dataset, healpix_map, lightcone, structure_collection); note the simulation
collection is identified structurally by `group_by_scope` (>1 header scope), not by a spec.

**Acceptance:** `SPEC.md` covers all five types and their identification signals; no code changes.

---

## Notes carried from exploration (correctness anchors for implementers)

- The spec registry is **self-contained in `io/specs.py`** — it does not use the plugin hook system
  (`plugins/hook.py`), which is reserved for extension logic that modifies core behavior, not core
  open-dispatch. Spec precedence is the order of the `SPECS` tuple; `match_spec` returns the first
  spec whose `matches()` is true (same "first-match wins" semantics `query()` had at
  `plugins/hook.py:60-74`, now with the order visible in one literal).
- `state_from_target(target, unit_convention, region, open_kwargs, index=None,
  metadata_group=None)` (`dataset/state.py:146`) is what the builders ultimately call —
  `metadata_group` selects which group's columns become `metadata_columns` (state.py:169-173).
  Keep it flowing unchanged in Phase 1.
- The "sorted for lockstep" invariant lives at `structure/io.py:404-409`; preserve it — the new
  path's determinism must not regress the write-side child ordering across ranks.
- No `test_discover.py`/`test_plan.py`/`test_specs.py` exist today; `test/test_distribute.py`
  currently covers only `partition_contiguous` (5 cases) and stays valid.
