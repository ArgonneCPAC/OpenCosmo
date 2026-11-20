# Metadata without materializing the catalog

Inspect schema and headers before choosing columns or numerical thresholds.
`get_data()`, `.data`, `.rows()`, and `objects()` are not metadata inspection.
Printing a Dataset representation can also read a small preview; use the
properties below rather than relying on `print(ds)` as a zero-I/O operation.

## Dataset and Lightcone column schema

Given an open `ds`:

```python
names = ds.columns  # list[str], current visible columns
units = ds.units  # dict[name, unit | None]
descriptions = ds.descriptions  # dict[name, str | None]
schema = {
    name: {"unit": units[name], "description": descriptions[name]} for name in names
}
required = {"fof_halo_mass", "sod_halo_cdelta"}  # example halo schema
missing = required.difference(names)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")
```

These describe the **current logical object**, not necessarily every on-disk
column. `select` changes visible names; expressions can supply inferred units;
`with_units` changes current units. Unitless columns normally have `None`.
Descriptions may be absent, including for columns created with `select`.

Metadata is not a substitute for validating data (sentinels, finite values,
empty samples). Only read values when that validation is necessary.

## Header exports: inspect keys, then values

```python
header = ds.header
exports = header.parameters
available = sorted(exports)
cosmology = exports["cosmology"]
simulation = exports.get("simulation")
redshift = exports.get("redshift")
step = exports.get("step")
```

`header.parameters` is a dictionary of **exported access paths**. Values can
be dictionaries, scalars, quantities, or transformed objects such as an Astropy
cosmology. Despite older docstrings, do not assume every value is a Pydantic
model or every block supports attribute access.

Equivalent access for an exported name:

```python
cosmology = ds.header.parameters["cosmology"]
assert cosmology is ds.header.cosmology
assert cosmology is ds.cosmology
```

`Dataset`, `Lightcone`, and some collections forward header exports through
attribute access. Prefer explicit `header.parameters` for discovery, optional
fields, or names that might conflict with API methods.

For HACC inputs, the simulation block is typically a dictionary:

```python
simulation = ds.header.parameters.get("simulation")
if isinstance(simulation, dict):
    simulation_fields = sorted(simulation)
    box_size = simulation.get("box_size")  # often a Quantity, not a float
    step_redshifts = simulation.get("step_zs")
```

Field availability depends on data origin and header schema. Simulation
parameters, cosmology-tool settings, map parameters, version information, and
other optional blocks are not guaranteed on every file. Do not hard-code a
field until you have checked the exports.

`header.file` exposes the validated file-parameter model, including origin,
data type, lightcone flag, step/redshift when supplied, and base file-unit
convention. This is different from current per-column units. `header.to_dict()`
returns a structured serialization payload, **not** the same view as
`header.parameters`; use the latter for scientific access.

## Cosmology and units

`ds.cosmology` is an Astropy cosmology object. Use its methods rather than
reimplementing cosmological distances or expansion functions:

```python
cosmology = ds.cosmology
redshift = ds.header.parameters.get("redshift")
if redshift is not None:
    distance = cosmology.comoving_distance(redshift)
h0 = cosmology.H0
omega_m = cosmology.Om0
sigma8 = cosmology.meta.get("sigma_8")
```

`Ob0` and extra metadata depend on the provided cosmology. Check availability
before computing a cosmic baryon fraction. Treat quantities as quantities;
convert with `.to(...)` / `.to_value(...)` when passing numbers externally.
Header quantities can reflect the current convention. Compare length scales
in compatible units, and read [with_units](api/with_units.md).

## Redshifts, steps, and collections

- **Snapshot Dataset:** `ds.header.parameters.get("redshift")` is the snapshot
  redshift when present; `step` is a simulation output identifier, not redshift.
- **Lightcone:** the aggregate is a redshift range, not a single snapshot.
  Discover keys and consult each child's header. Children can themselves be
  lightcones, so descend again for subtype/step nesting.

```python
child_metadata = {
    key: {
        "redshift": child.header.parameters.get("redshift"),
        "step": child.header.parameters.get("step"),
    }
    for key, child in lc.items()
}
```

  `with_redshift_range(z_low, z_high)` is the managed sample restriction;
  metadata redshift alone is not a row filter.
- **StructureCollection:** `collection.properties` lists property-column names;
  it is **not** the properties Dataset. Discover `collection.keys()` and access
  `collection[collection.dtype]` for its properties dataset (e.g.
  `"halo_properties"`). Consult each child for its columns/units/descriptions.
  Nested children remain collections, not tables.
- **SimulationCollection:** `collection.header` is keyed by simulation. Use
  `collection[name].header` for one member; do not assume identical cosmologies.
- **ParticleCollection:** inspect `particles[name].columns`, `.units`, and
  `.descriptions` per species; shared metadata lives on `particles.header`.
- **HealpixMap:** use `.columns`, `.descriptions`, `.nside`, `.nside_lr`,
  `.ordering`, `.full_sky`, `.pixels`, `.z_range`, and `.region`. Maps do not
  expose the ordinary `.units` property or support `with_units`; output arrays
  do not carry attached units. Header map parameters describe the product.
- **PowerSpectrum:** `.components`, `.steps`, `.redshifts`, `.k`, `.n_modes`,
  `.parameters`, and `.header` are available; opening already reads its spectra.

## Region and order

`ds.region` describes current spatial coverage; it is not a coordinate table.
`ds.sorted_by` reports the logical sort key (or `None`). A selected Dataset's
`len(ds)` is its current row count without requesting a whole table. Collection
`len` meanings differ; see the type table in `SKILL.md`.
