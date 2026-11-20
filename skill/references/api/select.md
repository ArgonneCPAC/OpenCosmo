# select

## Use case
Narrow a dataset/collection to a subset of columns, optionally creating new
columns in the same call. `select` is also how you **add columns**: pass `"*"`
to keep every existing column and add new ones alongside them (see
[Adding columns](#adding-columns)). **Always `select` (or `filter`) before expensive work**
— materialization reads selected outputs and their dependencies. Filters, sorting,
bounds, and evaluation can also read their required columns. Accepts wildcards:
`"fof*"` (prefix), `"*com*"` (substring).

## Baseline signature (Dataset, Lightcone)
```text
ds.select(*columns: str | Iterable[str],
          mode: Literal["local", "global"] = "global",
          **derived_columns: OpenCosmo column expression | np.ndarray | u.Quantity) -> Dataset
```
- Derived columns are `oc.col(...)` expressions (see [../expressions.md](../expressions.md)):
  `ds.select("fof_halo_mass", "*com*", fof_halo_px=oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx"))`
- Scalar reductions (`oc.col("x").mean()`) may be selected **only** alone (no
  mixing with column selections) — `ValueError` otherwise.
- `ValueError` if an exact column name is not present.
- `ValueError` if a new column reuses the name of an existing column; choose a
  new name.

## Adding columns
Use `select("*", new_name=...)` to keep all existing columns and add new ones.
The keyword name becomes the column name. Prefer this over
[evaluate](evaluate.md) whenever the new column is a closed-form expression of
existing columns — it stays lazy and unit-aware.
```python
import numpy as np
import opencosmo as oc

ds = oc.open("haloproperties.hdf5")
ds = ds.select("*", log_fof_halo_mass=oc.col("fof_halo_mass").log10())

# Several at once, or a raw array (length must equal len(ds)):
ds = ds.select(
    "*",
    fof_halo_px=oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx"),
    flag=np.ones(len(ds), dtype=bool),
)
```
- Expression columns are evaluated at materialization and **track unit
  conversions** (they transform correctly under [with_units](with_units.md)).
  Columns passed as `np.ndarray` or `u.Quantity` are stored as-is and do
  **not** change under unit transformations.
- `.log10()` on an expression normalizes by the column's unit.
- To keep only some existing columns, list them instead of `"*"`.
- Prerequisite columns must already exist; to build one new column from another,
  add them in separate `select` calls or reuse the expression object.

## Type-specific signatures
| Type | Signature notes |
|---|---|
| `StructureCollection` | `select(*select_args, mode="global", **select_kwargs)` — names/wildcards and derived-column expressions are routed automatically to whichever dataset contains their columns; target datasets explicitly with `dataset_name=[...]` or nested dicts (dot-paths like `"galaxies.star_particles"`). |
| `SimulationCollection` | `select(*args, **kwargs)` — delegates to each child. |
| `ParticleCollection` | `select(*select_args, mode="global", **select_kwargs)` — applies to every particle type. |
| `HealpixMap` | `select(*columns, **derived_columns)` — no `mode`. |

## Return shape
A new immutable object of the same type containing only the selected columns
(plus derived ones). Original is unchanged.

## Execution / MPI
Ordinary column/expression registration is logical (lazy); some scalar-dependent
operations can require computation. `mode` controls how
scalar reductions inside derived columns are computed under MPI: `"global"`
(default, cross-rank scalar) vs `"local"` (per-rank). No effect without MPI or
without scalar reductions.

## Examples
```python
import opencosmo as oc

ds = oc.open("haloproperties.hdf5")
centers = ds.select("fof_halo_mass", "fof_halo_center_*")
# sorted(centers.columns) contains mass and center_x/y/z (order not guaranteed).

# Derive from the original ds, which still contains the velocity column.
# Include 1/2 for the kinetic energy associated with this component.
kinetic = ds.select(
    "fof_halo_mass",
    kinetic=0.5 * oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx") ** 2,
)

# structure collection: automatic routing
collection = collection.select("fof_halo_mass", "gal_mass_star")

# add a derived column to every dataset that can compute it, keeping all columns
collection = collection.select(
    "*", gal_px=oc.col("gal_mass_star") * oc.col("gal_com_vx")
)
```

## Pitfalls
- Wildcards that match nothing are silently ignored; exact names that are
  missing raise `ValueError`.
- On `StructureCollection`, an exact name missing from *every* dataset raises;
  a name present in several datasets is selected from each (use explicit
  dataset kwargs to disambiguate).
- Do not call `get_data()` before `select` — you will materialize columns you
  don't need.
