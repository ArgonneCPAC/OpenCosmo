# Analysis: Distributed combine tools (`opencosmo.analysis.reduce` / `gather`)

Rank-ownership helpers for MPI runs. Both are importable without the yt extras:

```python
from opencosmo.analysis import reduce, gather  # safe: no yt/matplotlib needed
```

Under MPI these are **collectives: every rank must call them with the same
arguments**; in serial they just run locally. Default ownership: the result goes
to **root (rank 0)** and all other ranks receive `None`; pass `all=True` to return
it on every rank.

## `reduce` — combine computed results across ranks

```python
import numpy as np
import opencosmo as oc
from opencosmo.analysis import reduce

ds = oc.open("haloproperties.hdf5")


def halo_mass_function(fof_halo_mass, log_bins, box_size):
    log_mass = np.log10(fof_halo_mass)
    hist, _ = np.histogram(log_mass, log_bins)
    return hist / np.diff(log_bins) / box_size**3


result = reduce(  # None on non-root ranks by default
    ds,
    halo_mass_function,
    operation="sum",  # "sum" (default) | "prod" | "avg" (size-weighted)
    all=True,  # set True to get the result on every rank
    vectorize=True,
    format="numpy",  # forwarded to Dataset.evaluate
    log_bins=np.linspace(10.5, 15),
    box_size=ds.header.simulation["box_size"].value,
)
# result is a dict keyed by the function name: result["halo_mass_function"]
```

- Runs your function through `Dataset.evaluate` (`insert` is forced off) on each
  rank's local rows, then combines with `MPI.SUM` / `MPI.PROD` / weighted `SUM`.
  Extra `**kwargs` go to `evaluate`; a single ndarray return is unwrapped, a dict of
  ndarrays is reduced per key. Raises `ValueError` if ranks return different keys
  or different-length arrays.
- `SimulationCollection`: each simulation is reduced independently; result is a
  dict keyed by simulation name. `Lightcone`: per-step results are first combined
  with the requested operation (`combine_mode`) before the cross-rank reduction.
- Optional `plotting_function` / `plotting_kwargs` / `evaluate_kwargs`: the plotting
  function runs **only on the owning rank** (root unless `all=True`) and receives
  the result keys plus those kwargs as keyword arguments; its return value replaces
  the result.

## `gather` — concatenate columns onto one (or all) ranks

```python
from opencosmo.analysis import gather

table = gather(
    ds,
    columns=[
        "fof_halo_mass",
        "fof_halo_center_x",
        "fof_halo_center_y",
    ],  # or "all", single str, None
    format="numpy",  # "astropy" (default) | numpy | pandas | polars | arrow | jax
    all=False,
    fof_halo_px=oc.col("fof_halo_mass")
    * oc.col("fof_halo_com_vx"),  # derived columns via kwargs
)
# on rank 0: dict of arrays (or QTable for astropy); on other ranks: None
```

- Uses `select(...)` + `get_data("astropy")` locally, then gathers each column in
  rank order. With `columns=None`, only the derived (kwarg) columns are gathered.
  Raises `ValueError` if no columns were requested, or if ranks request different
  column sets.
- `plotting_function` receives one keyword argument per gathered column (plus
  `plotting_kwargs`) and runs only on the owning rank.
- `SimulationCollection`: gathers each simulation independently → dict keyed by
  simulation name.

## Choosing between local materialization and gather/reduce

- `ds.get_data()` under MPI returns **only the local rank's rows** — fine for
  per-rank work, wrong for a global result.
- Use `reduce` when you want a combined *computation* (histograms, sums,
  averages); use `gather` when you want the *rows themselves* on one rank (e.g. to
  plot on root). Avoid gathering full columns on huge datasets — prefer
  `binned_statistic`/`hist1d` with `mode="global"` or `reduce`.
