# Analysis: Plotting (`opencosmo.analysis.plotting`)

Matplotlib front ends for the functions in `opencosmo.analysis.statistics`.
Requires **matplotlib**, which is not a core dependency.

## Import (important)

```python
from opencosmo.analysis.plotting import hist1d, hist2d, binned_statistic
# or: import opencosmo.analysis.plotting as plotting
```

Import the plotting submodule by its full path. Plotting tools require the
optional visualization dependencies; install them before importing.

## Functions

Each wrapper mirrors its `statistics` counterpart and forwards extra kwargs
(`bins`, `bin_spacing`, `mode`, `q`, ...) straight through. Added arguments:

- `ax=None` — draw into an existing axis (enables multi-panel figures / overplots).
- `all=False` — under MPI, only **rank 0** draws by default; other ranks get
  `(None, None)`. The underlying statistic is still computed collectively on
  every rank. Set `all=True` to draw everywhere.
- `plot_kwargs=None` — passed to the matplotlib artist; caller values override
  defaults (`LogNorm(vmin=1)` for `hist2d`, `where="mid"` for `hist1d`,
  `linewidth=2` for `binned_statistic`).
- `dataset="halo_properties"` — member dataset when given a StructureCollection.

All three return `(fig, ax)` — or `(None, None)` on ranks that did not draw.

```python
import opencosmo as oc
from opencosmo.analysis.plotting import hist1d, hist2d, binned_statistic

ds = oc.open("haloproperties.hdf5").filter(oc.col("sod_halo_mass") > 1e13)

fig, ax = hist1d(ds, "sod_halo_mass", bins=30, differential="log")  # dN/dlog10 M
fig, ax = hist2d(
    ds, "sod_halo_mass", "sod_halo_cdelta", bins=80, plot_kwargs={"cmap": "magma"}
)
fig, ax = binned_statistic(ds, "sod_halo_cdelta", statistic="median")
# overplot a percentile band on the same axis:
binned_statistic(
    ds,
    "sod_halo_cdelta",
    statistic="quantile",
    q=0.84,
    ax=ax,
    plot_kwargs={"linestyle": ":", "color": "k"},
)
fig.savefig("concentration_mass.png")  # rank 0 only under MPI
```

- `hist1d(..., differential=None | "linear" | "log", yscale="log")`: `"linear"` →
  dN/dx, `"log"` → dN/dlog10(x) (requires unit-ful edges; raises `ValueError`
  otherwise).
- Axis labels, units, and log/linear scales are filled automatically from
  `default_params`; unknown
  columns fall back to the raw column name on a linear scale. When `bins` is an
  int and `bin_spacing` is omitted, the axis scale (`log`/`linear`) is used.

## Multi-panel figures

Pass the same `ax` to stack curves, and reuse one figure for several columns:

```python
from opencosmo.analysis.plotting import binned_statistic

fig, ax = binned_statistic(ds, "sod_halo_cdelta", statistic="median")
binned_statistic(
    ds, "sod_halo_cdelta", statistic="mean", ax=ax, plot_kwargs={"color": "r"}
)  # overplots on the same panel
```

## MPI plotting discipline

- Statistics are computed collectively on **all ranks**; `all=False` only controls
  who renders. Any figure manipulation (`savefig`, editing) must happen on rank 0.
- Root-only plotting after collectives is the default everywhere in
  `opencosmo.analysis`: `gather`/`reduce` run their `plotting_function` only on the
  rank that owns the result (root unless `all=True`), and these plotting wrappers
  return `(None, None)` on non-drawing ranks.
