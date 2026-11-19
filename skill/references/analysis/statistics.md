# Analysis: Statistics (`opencosmo.analysis.statistics`)

Managed, MPI-aware statistics over OpenCosmo datasets. Filter the sample before
calling these functions. Binned reductions may read required input columns, and
histograms materialize their input columns; account for this when working with
large datasets.

## Import

Import statistics from the explicit submodule:

```python
from opencosmo.analysis.statistics import binned_statistic, hist1d, hist2d
```

`from opencosmo.analysis import binned_statistic` may require the optional
yt/matplotlib packages. The explicit submodule import avoids that requirement.

## `binned_statistic` — per-bin reductions

```python
stat, edges = binned_statistic(
    ds,
    "sod_halo_cdelta",  # column to reduce
    bin_by="sod_halo_mass",  # default
    statistic="quantile",  # e.g. "mean" or "quantile"; see below
    bins=20,  # int -> bins+1 auto edges, or explicit edge list
    bin_spacing="linear",  # "linear" (default) or "log"; ignored for explicit edges
    dataset="halo_properties",  # member to use when ds is a StructureCollection
    mode="global",  # "global" (default) or "local" under MPI
    q=0.84,  # forwarded to the reduction, e.g. quantile
)
```

- Returns `(u.Quantity(list_of_bin_values), edges)`; `len(stat) == len(edges) - 1`.
- `statistic` accepts `"mean"`, `"std"`, `"var"`, `"min"`, `"max"`, `"median"`,
  `"sum"`, `"quantile"`, any of them prefixed with `"geometric_"` (reduction done in
  `log10` space, result raised back through `10**x`), or a callable
  `f(col, **kwargs) -> reduction expression`.
- Edges are determined from the requested sample, then each bin is filtered and
  reduced. Under MPI, `median`, `std`, `var`, and `quantile` may gather values to
  root; budget root memory. Do not assume constant memory.
- With an integer `bins`, edges span the min/max of `bin_by` in the requested
  mode (global across ranks by default, local if requested). Bins use `[low,
  high)`, including the last one, so a value exactly at the final edge is excluded.
- Columns listed in `default_params` get their `filter_bad` sentinel mask applied
  automatically (for `sod_halo_mass` the mask is `sod_halo_mass > 0`, which drops
  non-positive sentinel values such as `sod_halo_mass = -1` for unresolved halos).

## `hist1d` / `hist2d` — materialize columns, then histogram

```python
counts, edges = hist1d(ds, "sod_halo_mass", bins=30, bin_spacing="log")
h, x, y = hist2d(
    ds, "sod_halo_mass", "sod_halo_cdelta", bins=80, bin_spacing=("log", "linear")
)
```

- `hist1d(ds, column, bins=20, bin_spacing="linear", dataset=..., mode="global")`
  returns `(counts, bin_edges)`.
- `hist2d(ds, column_x, column_y, bins=100, bin_spacing="linear" | (sx, sy), ...)`
  returns `(h, x, y)` with `h.shape == (len(x)-1, len(y)-1)`; the first axis is
  `column_x`, so transpose `h` (`h.T`) before `pcolormesh`. `bin_spacing` may be a
  single string or a `(x, y)` pair.
- These functions read the full requested column(s) into memory on each rank
  before calling NumPy histogram functions. Filter/bound/select first to
  downselect. Under MPI with `mode="global"` the per-rank counts are summed,
  so every rank gets the global histogram.

## MPI behavior

- `mode="global"` (default): scalar reductions and histograms are combined across
  **all ranks**; every rank must execute the same calls. `mode="local"` gives
  each rank its own result. Both are no-ops in serial.
- Works identically in serial and parallel; same calling sequence either way.
- For a `StructureCollection`, pass `dataset="halo_properties"` (default) to pick
  the member dataset.

## Typical pattern

```python
import opencosmo as oc
from opencosmo.analysis.statistics import binned_statistic

ds = oc.open("haloproperties.hdf5")
ds = ds.filter(oc.col("sod_halo_mass") > 1e13)
c, edges = binned_statistic(
    ds, "sod_halo_cdelta", bin_spacing="log", statistic="geometric_std"
)
```
