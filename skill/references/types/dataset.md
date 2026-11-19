# Dataset

A `Dataset` (`opencosmo.Dataset`) is a single table of rows (e.g. halo or galaxy
properties at one snapshot, or one redshift step). `oc.open(path)` returns a
`Dataset` when the file contains exactly one `/data` group with no linked
particle datasets and no lightcone/map/power-spectrum structure.

```python
import opencosmo as oc

ds = oc.open("haloproperties.hdf5")
```

## Metadata access (dataset-dependent)

Column names, header fields, and units are properties of the specific file —
never hard-code them. Discover them at runtime:

```python
ds.columns  # list[str] — available column names
ds.descriptions  # dict[str, str | None] per column
ds.units  # dict[str, UnitBase | None] in the current convention
ds.header.parameters  # dict of header blocks; keys depend on origin + data type
list(
    ds.header.parameters
)  # e.g. "simulation", "cosmology", ... — inspect, don't assume
ds.header.simulation  # dot access works for any key in parameters
ds.region  # spatial Region the dataset is currently bounded to
ds.sorted_by  # column name or None
len(ds)  # number of selected rows (after filter/take/etc.)
```

`ds.some_name` also falls back to `header.parameters` lookup, so metadata and
columns share an attribute namespace.

## Operations

All transformations are immutable and lazy: they return new `Dataset` objects
and do not read data until materialization. Write the full query first, then
call `get_data()`.

```python
mass = oc.col("fof_halo_mass")  # use real column names from ds.columns

ds = (
    ds.filter(mass > 1e14)  # combine masks with & | ~
    .select(
        "fof_halo_mass",
        "fof_halo_center_*",  # wildcards allowed
        mass_x2=oc.col("fof_halo_mass") * 2,
    )  # derived columns as kwargs
    .sort_by("fof_halo_mass", invert=True)
)
data = ds.get_data()  # astropy QTable by default
data = ds.get_data("numpy")  # dict[str, ndarray]; also pandas/polars/arrow/jax
data = ds.get_data(wrap_single=True)  # force table even for one column
```

Key signatures:

- `filter(*masks, mode="global")` — masks built with `oc.col(...)`; see
  [../expressions.md](../expressions.md). `mode` controls how scalar reductions
  inside masks are computed under MPI; see [../mpi.md](../mpi.md).
- `select(*columns, mode="global", **derived_columns)` / `drop(*columns)` —
  positional names support wildcards; keyword args add expression columns.
- `take(n, at="start"|"end"|"random", mode="local")`,
  `take_range(start, end, mode="local")`, `take_rows(rows)`.
- `sort_by(column, invert=False)` — `sorted_by` then reports the column.
- `bound(region, select_by=None)` — restrict to a spatial region built with
  `oc.make_box` / `oc.make_cone` / `oc.make_skybox`.
- `with_units(convention=None, conversions=None, **columns)` — conventions:
  `"scalefree"`, `"comoving"`, `"physical"`, `"unitless"`; per-column unit
  overrides via kwargs.
- `with_new_columns(**columns, descriptions=None, ...)` — add columns from
  expressions or arrays without a Python callback.
- `evaluate(func, vectorize=False, insert=True, format="astropy", batch_size=-1)` —
  managed row/batch computation with automatic column selection; `insert=False`
  returns a dict instead of a new dataset. `evaluate_to_dataset` allows
  different-length output.
- `rows()` — iterate row dicts (select columns first for performance).
- `oc.write(path, ds)` — write the dataset or a transformed query.

A `Dataset` is **not** a mapping: there is no `ds["column"]`; use `select` +
`get_data`, or column expressions.

## Limitations

- `.data` is deprecated; use `get_data()`.
- Data is read at `get_data()` time (except filters/sorts/bounds/link reads,
  which may read what they need earlier) — always transform before materializing.
- Formats beyond astropy/numpy require the optional package (pandas, polars,
  arrow, jax) to be installed.

## Common API reference

See [../api/](../api/) for the shared operation reference and
[../expressions.md](../expressions.md) for building masks/derived columns.
