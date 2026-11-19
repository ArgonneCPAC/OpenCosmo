# get_data — materialize data

## Use case
The point where laziness ends: read the (transformed) data into memory. **Call
this only after all `filter`/`select`/`bound`/... operations** so you never
materialize data you don't need.

## Baseline signature (Dataset)
```text
ds.get_data(format: str = "astropy", unpack: bool = True,
            wrap_single: bool = False, **kwargs) -> QTable | Quantity | dict[str, ndarray] | ndarray
```
- `format`: `"astropy"` (default, QTable with units), `"numpy"` (dict of
  arrays, values in current convention, no unit objects), plus `"pandas"`,
  `"polars"`, `"arrow"`, `"jax"` — these need the corresponding optional
  package installed (checked at call time); polars/arrow copy data (slower).
- Single-column datasets return a bare Column/array/Series unless
  `wrap_single=True`, which forces the multi-column container.
- The deprecated `output=` kwarg is still accepted with a warning; use `format=`.

## Type-specific signatures / return shapes
| Type | Signature / return |
|---|---|
| `Lightcone` | `get_data(format="astropy", unpack=True, wrap_single=False, **kwargs)` — stacks children at materialization; `ignore_sort=True` kwarg skips sort-order restoration. Same formats as Dataset. |
| `SimulationCollection` | `get_data(format="astropy", wrap_single=False, unpack=True)` → `dict[str, data]` keyed by child name. Children must be Datasets/Lightcones. |
| `ParticleCollection` | `get_data(format="astropy", unpack=True, wrap_single=False, **kwargs)` → `dict[str, data]` keyed by particle type (e.g. `"dm_particles"`). |
| `HealpixMap` | `get_data(format="healsparse", nside_out=None)` — `"healsparse"` (default), `"healpix"` (NESTED full-sky arrays, uncovered pixels masked/UNSEEN), `"raw"` (`(pixels, {name: values})`). A single map column is unwrapped in healsparse/healpix formats, but raw is always a tuple. `nside_out` downsamples (must be ≤ current nside). Map arrays do not carry units; there is no `map.units` property. |
| `StructureCollection` | **No `get_data`** — use `collection["halo_properties"].get_data()`, or iterate `collection.objects()` / `halos()` / `galaxies()`. |
| `PowerSpectrum` | Eager object with its own `get_data(format="astropy", return_error=False)`; see [PowerSpectrum](../types/power_spectrum.md). |

## Execution / units / MPI
Reads only the currently selected columns and rows. Astropy output carries
units in the **current convention** (see [with_units](with_units.md)); numpy
output is in the current convention's units without unit objects. Under MPI each
rank materializes its own partition.

## Examples
```python
import astropy.units as u
import opencosmo as oc

ds = oc.open("haloproperties.hdf5")
ds = ds.filter(oc.col("fof_halo_mass") > 1e14 * u.Msun).select(
    "fof_halo_mass", "fof_halo_center_x"
)
table = ds.get_data()  # QTable, units attached (77 rows)
arrays = ds.get_data("numpy")  # dict[str, ndarray]
df = ds.get_data("pandas")  # requires pandas

single = ds.select(
    "fof_halo_mass"
).get_data()  # bare Quantity, NOT a table (single column)
table2 = ds.select("fof_halo_mass").get_data(wrap_single=True)  # forced QTable

lc_data = lc.get_data("numpy")  # stacked lightcone (lc from oc.open of step files)
maps = healpix_map.get_data(
    "healpix"
)  # dict of nested healpix arrays, e.g. {"ksz": ..., "tsz": ...}
props = collection["halo_properties"].get_data()
```

## Pitfalls
- `StructureCollection` has no `get_data` — indexing a dataset first is the
  intended pattern.
- Single-column standard dataset results are unwrapped by default (`wrap_single=True` to avoid). Maps have different unwrapping rules and no `wrap_single` option.
- A partial `"healpix"` map still allocates full-sky arrays; use `"raw"` or `"healsparse"` when appropriate.
- `.data` property is deprecated — use `get_data("astropy")`.
