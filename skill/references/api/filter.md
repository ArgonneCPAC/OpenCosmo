# filter

## Use case
Keep only rows satisfying boolean masks built from [column expressions](../expressions.md)
(`oc.col`). **Filter before any expensive operation** — the row selection is
pushed down so later reads only touch surviving rows. Snippets assume
`import astropy.units as u` and `import opencosmo as oc`.

## Baseline signature (Dataset, Lightcone)
```text
ds.filter(*masks: boolean column expressions,
          mode: Literal["local", "global"] = "global") -> Dataset
```
```python
massive = oc.col("fof_halo_mass") > 1e14 * u.Msun
fast = oc.col("fof_halo_com_vx") < -500 * u.km / u.s
ds = ds.filter(massive & fast)  # masks combine with &, |, ~
ds = ds.filter(oc.col("fof_halo_id").isin([1, 5, 42]))
```

## Type-specific signatures
| Type | Notes |
|---|---|
| `StructureCollection` | `filter(*masks, on_galaxies: bool = False, mode="global")` — masks apply to the halo/galaxy **properties** dataset; `on_galaxies=True` targets galaxies. Linked datasets are re-linked automatically. |
| `SimulationCollection` | `filter(*masks, **kwargs)` — delegates to each child. |
| `HealpixMap` | `filter(*masks)` — filters map pixels (rows = pixels). |
| `ParticleCollection` | **Unsupported** — no `filter` method. Bound a particle `Dataset` instead (`collection["gas_particles"].bound(...)`). |

## Return shape
New immutable object of the same type with only matching rows.

## Execution / units / MPI
- `filter` **does read data**: it materializes only the columns referenced by the
  masks to compute the row selection (this happens now, not at `get_data()`).
- `ValueError` if a mask references a missing column. A mask that matches **zero
  rows does not raise** — it returns an empty dataset (`len(ds) == 0`, and
  `get_data()` on it returns empty arrays).
- Masks compare in the dataset's current unit convention — call
  [with_units](with_units.md) first if your thresholds are in other units.
- MPI: `mode="global"` (default) combines scalar reductions (e.g.
  `oc.col("x").mean()` inside a mask) across all ranks so every rank filters
  against the same threshold; `"local"` uses per-rank scalars. No effect
  without MPI or without reductions in the mask.

## Masks can use columns you do not output

A mask may combine columns that are **not** part of your final output. Build the
expression from any columns of the dataset, filter, and only then project:

```python
import opencosmo as oc

ds = oc.open("haloproperties.hdf5")
gas_rich = oc.col("sod_halo_mass_gas") / oc.col("sod_halo_mass") > 0.1
selected = (
    ds.filter(gas_rich)  # uses two columns not in the output
    .select("fof_halo_mass")  # output only the mass
    .get_data("numpy")
)
```

The columns in a mask must still be present in the dataset you filter. If an
earlier `select` removed them, `filter` raises `MissingColumnError`:

```python
ds.select("fof_halo_mass").filter(gas_rich)  # raises: the gas columns were dropped
ds.filter(gas_rich).select("fof_halo_mass")  # works: filter first, then project
```

A column that does not exist in the dataset at all also raises
`MissingColumnError`.

## Examples
```python
import astropy.units as u
import opencosmo as oc

ds = oc.open("haloproperties.hdf5")
ds = ds.filter(oc.col("fof_halo_mass") > 1e14 * u.Msun)
print(len(ds))  # 77

# scalar-reduction threshold (global mean across ranks under MPI)
hot = oc.col("fof_halo_com_vx") > oc.col("fof_halo_com_vx").mean()
ds = ds.filter(hot)

# structure collection (masks apply to the properties dataset)
collection = collection.filter(oc.col("fof_halo_mass") > 1e14 * u.Msun)
```

## Pitfalls
- Calling `filter()` with no masks returns the dataset unchanged.
- Mask columns must exist in the dataset being filtered. Filter before `select`
  or `drop` removes them.
- Filtering to zero rows is allowed — you get an empty dataset, not an error.
- On `ParticleCollection` there is no `filter`; use spatial `bound` on the
  individual particle dataset for exact coordinate cuts.
