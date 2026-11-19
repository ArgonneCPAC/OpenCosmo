# evaluate — managed user-function computation

## Use case
Run a Python function over a dataset when the result is **not** a simple
algebraic expression (use [with_new_columns](with_new_columns.md) for those).
OpenCosmo selects only the columns the function needs.

## Dataset
```text
ds.evaluate(func, vectorize: bool = False, insert: bool = True,
            format: str = "astropy", batch_size: int = -1,
            allow_overwrite: bool = False, **evaluate_kwargs)
    -> Dataset | dict[str, np.ndarray]
ds.evaluate_to_dataset(func, vectorize=False, format="astropy", batch_size=-1,
                       allow_overwrite=False, **evaluate_kwargs) -> Dataset
```
- `func` takes either **column-named arguments** (`def f(fof_halo_mass, sod_halo_mass): ...`)
  or a single `data` argument receiving all selected columns
  (`def f(data): ...`). Extra inputs go through `**evaluate_kwargs` (kwarg names
  must not collide with column names — that raises `ValueError`). Return a dict
  `{name: array}` or a single array (named after the function).
- `vectorize=True`: full columns at once. `batch_size>0`: batches of rows
  (overrides `vectorize`). Default row-wise, one row at a time.
- **`insert=True` (default): LAZY** — registers a lazily computed column; your
  function is called once on a single sample row at call time to check the
  output names, units, and shape. The remaining rows are computed when data is
  materialized. With `insert=False` the function runs **eagerly** over the
  whole dataset right away and a `dict[str, np.ndarray | Quantity]` is
  returned (no sample-row call).
- Computed outputs are stored as numpy arrays or Quantities and are **not
  unit-aware**: they do **not** change under [with_units](with_units.md)
  conversions. Use `with_new_columns` expressions when you need unit-aware
  lazy columns.
- Row-wise array outputs: with `insert=True` each row's array is one entry of
  a (possibly multidimensional) column (stacked); with `insert=False` per-row
  arrays are concatenated.
- `format`: "astropy" (default), "numpy", "pandas", "polars", "arrow", "jax"
  — the format of the *input* data passed to `func`. With `format="numpy"`
  values arrive as raw ndarrays (no units). Use the "astropy" default when your
  function needs unit-carrying columns.
- `evaluate_to_dataset`: always evaluates **immediately** and returns a **new
  in-memory Dataset containing only the output columns** (lengths need not
  match the input). Evaluated in the file's baseline convention, then the
  current convention is restored; header retained, spatial information is not.
  Outputs must be numpy arrays or Quantities of equal length.

## Type-specific signatures
| Type | Notes |
|---|---|
| `Lightcone` | `evaluate(func, vectorize=False, insert=True, format="astropy", batch_size=-1, allow_overwrite=False, combine_mode="concatenate"|"sum"|"prod"|"avg", **kw)` — children are combined at materialization; `combine_mode` only affects `insert=False` results. |
| `StructureCollection` | `evaluate(func, dataset: str | None = None, format="astropy", insert=True, allow_overwrite=False, **kw)` — runs **per structure** over linked datasets. Args may be dataset names (`def f(halo_properties, gas_particles): ...`, with per-dataset column lists as kwargs like `gas_particles=["mass"]`) or **column names**, whose dataset is inferred when unique (`def f(fof_halo_mass, mass): ...`); ambiguous columns raise `ValueError` — narrow with `with_datasets([...])`. Results are added to `halo_properties` (or galaxy properties). `evaluate_on_dataset(func, dataset=None, vectorize=False, format="astropy", insert=True, batch_size=-1, allow_overwrite=False, **kw)` — runs on the **whole named dataset** (default `halo_properties`), NOT per structure. Default is row-wise over that dataset (`vectorize=False`); set `vectorize=True` for whole columns. |
| `SimulationCollection` | `evaluate(func, datasets=None, format="astropy", vectorize=False, insert=False, ...)` — note `insert` defaults to **False** here. |
| `ParticleCollection` | Only `evaluate_on_dataset(func, dataset: str, vectorize=True, format="astropy", insert=True, batch_size=-1, ...)` — note `vectorize` defaults to **True**; `dataset` is required. No plain `evaluate`. |
| `HealpixMap` | `evaluate(func, format="numpy", vectorize=False, insert=True, **kw)` — note `format` defaults to `"numpy"`. |

## Return shape
`insert=True` → new object of the same type with the new column(s);
`insert=False` → `dict[str, np.ndarray | Quantity]`. Computed columns do
**not** change under [with_units](with_units.md) unit conversions.

## Examples

Prerequisite for all snippets below: set `path` to a real OpenCosmo
halo-properties file.

```python
import numpy as np
import astropy.units as u
import opencosmo as oc

path = "haloproperties.hdf5"
ds = oc.open(path)
```

```python
def baryon_bias(sod_halo_mass_gas, sod_halo_mass, cosmology):
    f_gas = sod_halo_mass_gas / sod_halo_mass
    return {"sod_halo_baryon_bias": f_gas / (cosmology.Ob0 / cosmology.Om0)}


sub = ds.take(10, at="start")
ds2 = sub.evaluate(baryon_bias, vectorize=True, cosmology=ds.cosmology)  # lazy insert
# ds2 has a new "sod_halo_baryon_bias" column.

# Return values instead of inserting: this computes immediately.
out = sub.evaluate(baryon_bias, vectorize=True, insert=False, cosmology=ds.cosmology)
```

Use `evaluate` when the computation needs an external input that an expression
cannot reference, such as a lookup table interpolated at each row. Here the
observed gas fraction within R500c is compared with a reference relation
(`np.interp` over a tabulated mass grid):
```python
ref_mass = np.array([1e13, 3e13, 1e14, 3e14, 1e15])  # solMass
ref_fgas = np.array([0.08, 0.10, 0.12, 0.13, 0.14])


def gas_fraction_residual(sod_halo_M500c, sod_halo_MGas500c, ref_mass, ref_fgas):
    fgas = sod_halo_MGas500c / sod_halo_M500c
    expected = np.interp(sod_halo_M500c, ref_mass, ref_fgas)
    return {"fgas_residual": fgas - expected}


# Return the values without changing the dataset.
resid = sub.evaluate(
    gas_fraction_residual,
    vectorize=True,
    insert=False,
    format="numpy",
    ref_mass=ref_mass,
    ref_fgas=ref_fgas,
)

# Or keep a lazy column that is computed when the data is requested.
ds_e = sub.evaluate(
    gas_fraction_residual,
    vectorize=True,
    format="numpy",
    ref_mass=ref_mass,
    ref_fgas=ref_fgas,
)
values = ds_e.select("fgas_residual").get_data("numpy")
# The evaluated column is not unit-aware: it does not change with with_units().

# Prefer an expression whenever one can express the calculation. For example,
# log10 of a mass column is a column expression, not an evaluate callback.
# Mask non-positive sentinel masses first, since their log is undefined:
positive = sub.filter(oc.col("sod_halo_M500c") > 0)
log_mass = positive.with_new_columns(log_m=oc.col("sod_halo_M500c").log10())


# New dataset from arbitrary-length output (evaluated immediately).
def count_rows(data):
    return {"n": np.full(1, len(data["fof_halo_tag"]))}


per = sub.evaluate_to_dataset(count_rows, vectorize=True)
# per contains only "n" and has one output row.
```

### Linked managed computation: gas-particle sum / halo mass

Prerequisite: the paths below identify related halo-properties and particle
files. Narrow the collection when a column name appears in multiple datasets.
```python
coll = oc.open("haloproperties.hdf5", "haloparticles.hdf5").take(20, at="start")

# Inferred unique columns after narrowing to two datasets.
narrow = coll.with_datasets(["halo_properties", "gas_particles"])


def gas_to_total_mass(fof_halo_mass, mass):  # mass inferred from gas_particles
    return np.sum(mass) / fof_halo_mass


res = narrow.evaluate(gas_to_total_mass, format="numpy")
col = res["halo_properties"].select("gas_to_total_mass").get_data("numpy")


# Explicit dataset arguments on the full collection, insert=False.
def explicit(halo_properties, gas_particles):
    return np.sum(gas_particles["mass"]) / halo_properties["fof_halo_mass"]


exp = coll.evaluate(
    explicit,
    format="numpy",
    insert=False,
    halo_properties=["fof_halo_mass"],
    gas_particles=["mass"],
)

# Whole-dataset (not per-structure) evaluation of the same residual, on the
# halo properties dataset only.
res2 = coll.evaluate_on_dataset(
    gas_fraction_residual,
    dataset="halo_properties",
    ref_mass=ref_mass,
    ref_fgas=ref_fgas,
)
```

## Pitfalls
- Function returning `None` with `insert=True` raises `ValueError`.
- A **bare (non-dict) return** is named after the function, so a `lambda`
  raises `TypeError("Lambda functions cannot name output columns")` — use a
  `def` or return a dict.
- Output keys must not collide with existing columns unless
  `allow_overwrite=True`.
- Unknown required arguments (not columns, not kwargs) raise `ValueError`.
- With `insert=True` the function is called once on a single sample row at call
  time; keep single-row calls cheap. The remaining rows are computed lazily at
  materialization.
- `evaluate` is **not** a bypass of lazy evaluation: `insert=True` stays lazy
  (but not unit-aware); only `insert=False` is eager.
- Prefer managed ops (`filter`, `select`, expressions via `with_new_columns`)
  over `evaluate` when possible.
