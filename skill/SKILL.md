---
name: opencosmo-library
description: >-
  Write and debug Python analysis using OpenCosmo cosmological simulation data:
  catalogs, lightcones, linked halos/galaxies/particles, snapshot particles,
  HEALPix maps, and power spectra. Use for opencosmo imports, lazy column
  expressions, cosmology-aware units and metadata, managed evaluation, MPI,
  and supported statistics or visualization integrations.
metadata:
  opencosmo-version: "1.4.7"
---

# OpenCosmo analysis

OpenCosmo reads structured HDF5 simulation products and exposes their units,
cosmology, spatial indices, and relationships through immutable Python objects.
Use only the public API (`import opencosmo as oc`); do not read raw HDF5 datasets directly as a workaround.

## Verification (always run first)

Always check OpenCosmo is installed, and that its version matches the version in this skill metadata. Do the following:


```bash
python -c "import opencosmo; print(opencosmo.__version__)"
```

If the major or minor version of the library does not match the the version in the skill metadata, warn the user about the mismatch but continue.


## Workflow: plan the query before materializing

1. **Open and identify the abstraction.** `oc.open(*paths)` discovers the layout;
   file count or filenames alone do not determine the returned type.
2. **Inspect metadata, not row data.** Discover available columns, descriptions,
   units, parameter blocks, cosmology, redshifts, and child keys. Do not call
   `.data`, `.rows()`, or `get_data()` just to discover the schema.
3. **Choose units and restrict the science sample.** Apply spatial/redshift
   bounds and filters before expensive derived calculations. Numeric thresholds
   use current units. Do not move a filter across a reduction if that changes
   the statistical population.
4. **Keep only needed inputs/children.** Use `select`, `with_datasets`, or
   `objects(data_types=...)`. Keep inputs needed by later expressions/evaluation;
   finish by selecting only requested outputs. Do not default to `select("*")`.
5. **Use managed computation.** Prefer `oc.col` arithmetic, masks, and scalar
   reducers. Use `evaluate` for custom functions, structure-wise evaluation for
   linked data, and built-in statistics when applicable. Avoid materializing
   whole tables to do equivalent NumPy/pandas work manually.
6. **Materialize at the boundary.** Call `get_data` only for the final result
   needed by plotting, an external algorithm, or the caller. `oc.write(path,
   query)` can write a transformed object directly; no preliminary `get_data`.
7. **Execute and check the result.** Verify shape, units, row alignment, empty
   samples, and optional dependencies on a small representative query first.

NumPy inside a managed callback is appropriate when expressions cannot express
that computation. This is not a prohibition on external scientific algorithms.

## Core principles

- **Immutable transformations:** `ds = ds.filter(mask)`; ignoring the returned
  object does not change `ds`. Transformations never modify the original; treat
  returned arrays as read-only snapshots.
- **Lazy materialization, not zero I/O:** column expressions are evaluated
  lazily; filters read their required columns, exact bounds read boundary
  coordinates, and sorting reads the sort key when needed. `evaluate(insert=True)`
  can call your function once on a sample row at registration. `PowerSpectrum`
  is an eager exception.
- **Units and cosmology are part of the query:** files normally open in the
  comoving convention. Expression columns participate in unit inference and
  conversion; supplied arrays and arbitrary evaluated outputs do not acquire
  cosmological conversion rules. Inspect units before numeric thresholds.
- **Preserve relationships:** query a `StructureCollection` to keep properties
  and linked rows aligned. Use `SimulationCollection.match` when matching
  metadata exists, rather than joining by row position or filtering each member
  independently. Lightcone scalar reductions can span the stacked children.

## Identify the object

| Type | Meaning | Key distinctions |
|---|---|---|
| `Dataset` | One row/column dataset | `len` is selected rows; standard query API |
| `Lightcone` | Related redshift-step datasets | Dataset-like stacked results; keys are steps/subtypes; `len` is total rows |
| `StructureCollection` | Properties plus linked detail, possibly nested | `len` is structures; no collection-level `get_data`; iterate `objects()` |
| `SimulationCollection` | Same-kind children from different simulations | `len` is simulations; child-keyed results; matching changes row operations |
| `ParticleCollection` | Snapshot datasets by particle species | `len` is species; restricted API; collection bounds are index-snapped |
| `HealpixMap` | Map values with pixel coverage | `len` is pixels; map output formats; unit conversion unsupported |
| `PowerSpectrum` | Spectra by component and step | Eager opening; separate selection API; not MPI-distributed |

Mapping iteration yields **keys**, not rows. Discover `keys()` rather than
assuming filenames become keys. Do not assume every type has `filter`, `take`,
`evaluate`, `with_units`, or the same `get_data` format.

## Minimal catalog query

Example columns below are HACC halo columns; validate them for your input.
Set `path` to a real OpenCosmo halo-properties file.

```python
import opencosmo as oc

with oc.open(path) as ds:
    assert isinstance(ds, oc.Dataset)
    needed = {"fof_halo_mass", "sod_halo_cdelta"}
    missing = needed.difference(ds.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    query = (
        ds.with_units("comoving")
        .filter(oc.col("fof_halo_mass") > 1e14)
        .select("fof_halo_mass", "sod_halo_cdelta")
        .take(1000, at="start")
    )
    table = query.get_data(format="astropy", wrap_single=True)
```

`take` defaults to **random**, not head. Specify `at` when intent matters.
`get_data` normally unwraps single-column results; use `wrap_single=True` for a
stable standard table/dictionary container. Scalar selections and map results
have their own shapes.

## Metadata first

For a `Dataset` (also many fields on a `Lightcone`):

```python
names = ds.columns  # visible column names; no table read
units = ds.units  # name -> current unit or None
labels = ds.descriptions  # name -> description or None
parameters = ds.header.parameters  # available named metadata exports
cosmology = ds.cosmology  # Astropy cosmology, when exported
redshift = parameters.get("redshift")
simulation = parameters.get("simulation")
```

Exports depend on origin/type; parameter blocks are not all the same kind of
object. Inspect keys before accessing a simulation field. Use child headers for
per-step/per-simulation values, not one aggregate redshift for a lightcone.
Read [metadata](references/metadata.md) for concrete access patterns.

## Read the relevant reference before writing specialized code

Paths below are relative to this `SKILL.md`. Load only those needed for the task.

### Common API

- [open](references/api/open.md): discovery, related files, lifetime, opening modes.

- [select](references/api/select.md): projection, wildcards, derived names, routing.
- [drop](references/api/drop.md): removal and type-specific signatures.
- [filter](references/api/filter.md): masks, units, reduction thresholds, structures.
- [take / take_range / take_rows](references/api/take.md): samples, ranges, ordering.
- [sort_by](references/api/sort_by.md): sort keys and top-N queries.
- [bound and region constructors](references/api/bound.md): geometry and exactness.
- [with_units](references/api/with_units.md): conventions and explicit conversions.
- [with_new_columns](references/api/with_new_columns.md): arrays, descriptions, routing.
- [evaluate](references/api/evaluate.md): callbacks, vectorization/batching, linked
  evaluation, insertion versus returned results, `evaluate_to_dataset`.
- [get_data](references/api/get_data.md): materialization, formats, unwrapping, units.
- [write](references/api/write.md): direct query writing and optional Parquet.
- [expressions](references/expressions.md): arithmetic, masks, scalar reductions,
  normalization, dependency binding, unit-aware scientific calculations.

### Types and relationships

- [Dataset](references/types/dataset.md): ordinary catalog queries.
- [Lightcone](references/types/lightcone.md): redshift/pixel pruning, stack-wide
  statistics, attached-map cutouts.
- [StructureCollection](references/types/structure_collection.md): linked
  halo/galaxy/particle analysis; read this before per-object computation.
- [SimulationCollection](references/types/simulation_collection.md): matching
  and reference-driven filtering; `clear_match` does not restore removed rows.
- [ParticleCollection](references/types/particle_collection.md): species routing
  and index-snapped spatial bounds.
- [HealpixMap](references/types/healpix_map.md): coverage, NESTED pixels,
  resolution changes, map cutouts and output formats.
- [PowerSpectrum](references/types/power_spectrum.md): components, steps, errors.

### MPI and analysis integrations

- [MPI](references/mpi.md): **read before parallel code**; collective participation,
  local/global defaults, empty ranks, distributed I/O and memory risks.
- [Distributed gather/reduce](references/analysis/distributed.md): collect
  final results or compute across ranks without gathering a whole catalog.
- [Statistics](references/analysis/statistics.md): managed histograms and binned
  statistics; numerical outputs and bin conventions.
- [Plotting](references/analysis/plotting.md): matplotlib wrappers and rank ownership.
- [yt visualization](references/analysis/yt.md): optional dependencies and conversion.
- [Diffsky integration](references/analysis/diffsky.md): supported catalogs,
  host relationships, metadata and version-compatible dependencies.

## Before delivering code

- Did the query retain the intended sample, units, and reduction population?
- Could OpenCosmo do arithmetic/reductions/linking that the code does manually?
- Are only necessary columns/children read, and materialization justified?
- Are callbacks pure and valid on a sample row/batch, including empty inputs?
- Are return shapes, matched rows, and collection-specific limitations checked?
- Under MPI, do **all ranks** enter collectives, even if only root uses the result?

Written for OpenCosmo 1.4.7; check installed signatures if your version differs.
