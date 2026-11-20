# open — `oc.open`

## Use case
Entry point for reading OpenCosmo HDF5 files. The returned object type is driven
by the **structure of the files**, not by the number of paths: a single
multi-group file can produce a collection, and several redshift-step files
usually produce one `Lightcone`.

```python
import opencosmo as oc

ds = oc.open("haloproperties.hdf5")  # -> Dataset
collection = oc.open(
    "haloproperties.hdf5", "haloparticles.hdf5"
)  # -> StructureCollection
lc = oc.open(
    "step_600/haloproperties.hdf5", "step_601/haloproperties.hdf5"
)  # -> Lightcone
```

## Signature
```text
oc.open(*files: str | Path, mpi_mode: str = "spatial", **open_kwargs: bool)
```

- `*files`: one or more paths (or a single list of paths). Paths are sorted for consistent discovery; order does not affect the result.
- `mpi_mode`: `"spatial"` (default) — every rank opens every file and partitions
  rows spatially. `"redshift"` — lightcone files are distributed across ranks by
  redshift step; empty ranks still hold a full-schema, zero-length dataset so
  `select`/`filter`/reductions behave identically on every rank. Nested
  (Diffsky) lightcones fall back to spatial. No-op without MPI. `ValueError` if
  the value is invalid or `"redshift"` is used on a non-lightcone.
- `**open_kwargs`: boolean flags that restrict which datasets load when
  multiple files are combined (ignored for a single single-dataset file).

## Returns
One of: `Dataset`, `Lightcone`, `StructureCollection`, `SimulationCollection`,
`HealpixMap`, `ParticleCollection`, `PowerSpectrum`. See [../types/](../types/).

`PowerSpectrum` is read eagerly.

## Execution
Opening is lazy: column data is not read until requested by an operation or
materialized with `get_data()`.

## Examples
```python
import opencosmo as oc

# Single dataset
ds = oc.open("galaxyproperties.hdf5")

# Related files -> high-level collection
halos = oc.open("haloproperties.hdf5", "haloparticles.hdf5")

# Lightcone under MPI with redshift-step distribution
lc = oc.open(*steps, mpi_mode="redshift")
```

## Pitfalls
- Passing several files that are *not* a recognized combination raises an error;
  the library always returns a single object, never a list.
- `mpi_mode="redshift"` on snapshots/structure collections is a `ValueError`.
- Do not assume one file == one `Dataset`; inspect the returned type
  (`isinstance` or `type(obj).__name__`).
