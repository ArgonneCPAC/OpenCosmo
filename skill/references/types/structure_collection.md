# StructureCollection

A `StructureCollection` (`opencosmo.StructureCollection`) pairs a **properties
dataset** (e.g. `halo_properties` / `galaxy_properties`) with one or more
**linked datasets** (e.g. `dm_particles`, `gas_particles`) whose rows are
associated to property rows via `/data_linked` start/size metadata. `oc.open`
returns one when it finds a properties group with linked targets, possibly
across multiple files:

```python
import opencosmo as oc

collection = oc.open("haloproperties.hdf5", "haloparticles.hdf5")
```

Collections can nest (e.g. a galaxy `StructureCollection` inside a halo
collection) and can wrap lightcone steps.

## Keys, objects, len

- Mapping keys are **dataset names** (strings), e.g.
  `collection.keys()` — the properties dataset is listed **last**.
  `collection[name]` returns a `Dataset`, or a nested `StructureCollection`.
- `len(collection)` is the number of structures (rows of the properties
  dataset). It is **not** the number of datasets.
- `collection.properties` — list of column names in the properties dataset.
- `collection.dtype` — `"halo_properties"` or `"galaxy_properties"`;
  `collection.simulation` — the simulation parameter block from the header.

## objects(): heterogeneous per-object mapping

`objects()` iterates **one object at a time** and yields a heterogeneous dict:

```python
for halo in collection.objects():  # halos() / galaxies() are aliases
    props = halo["halo_properties"]  # dict of scalar values for this object
    dm = halo["dm_particles"]  # a full opencosmo.Dataset with this
    gas = halo["gas_particles"]  # object's rows (lazily linked)
```

The properties entry is a plain `dict`; every linked entry is a `Dataset`.
Use `objects(data_types=[...])` or `collection.with_datasets([...])` to limit
which datasets are touched. `ignore_empty=True` (default) skips objects with no
linked rows. Links are rebuilt automatically after row operations
(filter/take/sort), so object iteration stays aligned with properties.

## Metadata access

```python
collection.header.parameters  # dict; keys are dataset-dependent — inspect
list(
    collection.header.parameters
)  # e.g. "simulation", "cosmology", "halo_properties"...
collection.keys()
collection["dm_particles"].columns  # per-dataset columns
collection.region
collection.sorted_by
```

## Operations

Row ops (`filter`, `take`, `take_range`, `take_rows`, `sort_by`) apply to the
properties dataset and **rebuild the links** so linked datasets stay aligned.
Column ops are routed per dataset:

- `select(*columns, mode="global", **per_dataset_columns)` — kwargs keyed by
  dataset name (dot paths like `"galaxies.star_particles"` for nested) select
  or derive columns in that dataset.
- `drop(...)`, `with_units(..., **dataset_conversions)` — same per-dataset
  routing.
- `bound(region)` works on **any** structure collection (snapshot-backed
  included); `cone_search` / `box_search` / `pixel_search` /
  `with_redshift_range` are valid only on lightcone-backed collections.
- `evaluate(func, dataset=None, insert=True, format="astropy", **kwargs)` —
  the function takes **dataset names** (or individual column names) as
  arguments: properties columns arrive as scalars, linked-dataset columns as
  arrays of that object's rows. With `dataset="name"` (or a dot path) it runs
  on one dataset only (`evaluate_on_dataset` is the explicit form). Results
  insert into the properties dataset by default.

```python
import numpy as np


def gas_fraction(fof_halo_mass, mass):
    return np.sum(mass) / fof_halo_mass
```

(`mass` here is the linked particle dataset's per-row mass column; the
properties columns arrive as scalars, linked-dataset columns as arrays.)

```python
collection = collection.with_datasets(["halo_properties", "gas_particles"])
collection = collection.filter(oc.col("fof_halo_mass") > 1e14)  # rebuilds links
collection = collection.evaluate(gas_fraction)  # inserts column
```

## Limitations

- **`get_data` is NOT supported on `StructureCollection`** — call it on
  individual datasets (`collection["halo_properties"].get_data()`) or iterate
  `objects()`. (Child `Dataset`s support it; the collection does not.)
- Column names shared by multiple datasets are ambiguous in `evaluate` —
  narrow with `with_datasets` or pass dataset-name arguments.
- `bound(region)` works on snapshot-backed collections too. The lightcone-only
  methods (`with_redshift_range`, `cone_search`, `box_search`, `pixel_search`)
  raise `TypeError` on snapshot-backed collections.

## Common API reference

See [../api/](../api/) and [../expressions.md](../expressions.md); MPI notes in
[../mpi.md](../mpi.md) (structure collections distribute across ranks).
