# Lightcone

A `Lightcone` (`opencosmo.Lightcone`) holds simulation data projected onto the
plane of the sky: one or more redshift-step `Dataset`s (optionally nested by
subtype) that stack into a single table at materialization time. `oc.open(...)`
returns a `Lightcone` when the input files' `/data` groups are all lightcone
groups of one data type (e.g. several redshift steps of `halo_properties`).

```python
import opencosmo as oc

lc = oc.open("haloproperties_600.hdf5", "haloproperties_601.hdf5")
```

## Keys, values, len

- Mapping interface over **redshift steps**: `lc.keys()` are step identifiers
  (`int` or `str`), `lc[key]` returns a `Dataset` (plain lightcone) or a nested
  `Lightcone` (nested by subtype), `len(lc)` is the **total number of rows
  across all steps** (`sum(len(step) for step in lc.values())`), not the number
  of steps.
- Iterating yields keys; `lc.items()` / `lc.values()` give the children.

## Metadata access

Metadata fields are dataset-dependent; discover them, don't hard-code:

```python
lc.columns  # columns available on every child
lc.header.parameters  # header of the first remaining step;
list(lc.header.parameters)  # inspect available blocks
lc.header.lightcone  # lightcone block, e.g. z_range
lc[600].header.parameters["redshift"]  # each step keeps its own header/redshift
lc.units, lc.descriptions, lc.region, lc.sorted_by
lc.get_pixels(nside=64)  # HEALPix pixels (nested) covered by the lightcone
```

## Stacking and whole-lightcone reductions

- `get_data()` **stacks** all remaining children into one table (ordered by
  step, then subtype). All row/column operations apply per child, so build the
  full query (filter/select/sort/take) before `get_data()`.
- Expressions containing a **scalar reduction** (e.g.
  `oc.col("fof_halo_mass").mean()`) or depending on an existing lightcone
  column are computed over the *whole stacked lightcone*, not per child — and
  lazily, seeing only rows surviving all prior operations. Elementwise
  expressions are applied to each child.
- `evaluate(..., insert=False, combine_mode="concatenate"|"sum"|"prod"|"avg")`
  combines per-child results; `"avg"` weights by per-child row counts, and all
  non-`concatenate` modes require equal result lengths. `combine_mode` has no
  effect when `insert=True`.

```python
# Redshift of each row relative to the global mean redshift (one scalar over the stack)
lc = lc.select(
    "redshift",
    zscore=(oc.col("redshift") - oc.col("redshift").mean()) / oc.col("redshift").std(),
)
```

## Unique operations

- `with_redshift_range(z_low, z_high)` — prune steps by redshift; may leave a
  single step (still a `Lightcone`).
- Spatial queries prune by HEALPix pixel: `bound(region)`, `cone_search(center,
  radius)`, `box_search(p1, p2)`, `pixel_search(pixels, nside=64)`.
- `cutouts(size, npix=64)` — yield `(metadata, fits.HDUList)` cutouts around
  each object; requires a HEALPix map opened alongside the lightcone (raises
  `ValueError` otherwise).
- Standard ops mirror `Dataset` but return `Lightcone`:
  `filter`, `select`/`drop`, `take`/`take_range`/`take_rows`, `sort_by`,
  `with_units`, `evaluate`, `rows`, `get_data`.

Empty results keep structure: filtering to zero rows keeps all steps
(`len(lc) == 0`) rather than erroring.

## Limitations

- `get_data` supports the same formats as `Dataset`, but stacking means the
  whole lightcone is materialized at once — filter/select first.
- A nested step with one remaining subtype is still a `Lightcone`; row
  operations on a nested lightcone drop empty subtypes.

## Common API reference

See [../api/](../api/) for shared operations, [../expressions.md](../expressions.md)
for expressions and whole-lightcone scalar reductions, [../mpi.md](../mpi.md) for MPI behavior of
lightcones (redshift-mode distribution, `mode="local"|"global"`).
