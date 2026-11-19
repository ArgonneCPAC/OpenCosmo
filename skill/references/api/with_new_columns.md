# with_new_columns

## Use case
Add columns that are algebraic combinations of existing columns (built with
`oc.col`, see [../expressions.md](../expressions.md)), raw numpy arrays, or
astropy quantities. Prefer this over [evaluate](evaluate.md) whenever the new
column is a simple expression of existing columns — it stays lazy and
unit-aware.

## Baseline signature (Dataset, Lightcone)
```text
ds.with_new_columns(descriptions: str | dict[str, str] | None = None,
                    allow_overwrite: bool = False,
                    mode: Literal["local", "global"] = "global",
                    **new_columns: OpenCosmo column expression | np.ndarray | u.Quantity) -> SameType
```
The keyword name becomes the column name.
```python
log_mass = oc.col(
    "fof_halo_mass"
).log10()  # expression .log10() normalizes by the column's unit
ds = ds.with_new_columns(log_fof_halo_mass=log_mass)
ds = ds.with_new_columns(
    fof_halo_mass=log_mass, allow_overwrite=True
)  # replace in place
```

## Type-specific signatures
| Type | Notes |
|---|---|
| `StructureCollection` | `with_new_columns(dataset: str, descriptions=None, allow_overwrite=False, mode="global", **new_columns)` — the target dataset is required; use dot paths for nested (`"galaxies.star_particles"`). `ValueError` if dataset missing. |
| `SimulationCollection` | `with_new_columns(*args, datasets=None, descriptions=None, allow_overwrite=False, **new_columns)` — `datasets=` restricts which children get the column. |
| `HealpixMap` | `with_new_columns(descriptions=None, **columns)` — **no** `allow_overwrite`/`mode`. |
| `ParticleCollection` | **Unsupported** — no `with_new_columns`. |

## Return shape
New immutable object of the same type with the columns added.

## Execution / units / MPI
Lazy: expression columns are evaluated at materialization and **track unit
conversions** (they transform correctly under [with_units](with_units.md)).
Columns passed directly as `np.ndarray` or `u.Quantity` are stored as-is and
do **not** change under unit transformations. `mode` controls scalar reductions
nested in expressions under MPI (`"global"` default vs `"local"`); no effect
otherwise. `ValueError` if the name exists and `allow_overwrite=False`.

## Examples
```python
import numpy as np
import opencosmo as oc

ds = oc.open("haloproperties.hdf5")
ds = ds.with_new_columns(
    fof_halo_px=oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx"),
    descriptions="x-momentum",
)
ds = ds.with_new_columns(
    flag=np.ones(len(ds), dtype=bool)
)  # raw array: unitless, never converted

collection = collection.with_new_columns(
    "dm_particles", pe=oc.col("phi") * oc.col("mass")
)
```

## Pitfalls
- Mixing an `oc.col` expression with arrays is fine, but remember only
  expression columns are unit-aware.
- Overwriting an existing column requires `allow_overwrite=True`.
- Use `evaluate` instead when the computation needs a Python function, not a
  closed-form expression.
