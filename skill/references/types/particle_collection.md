# ParticleCollection

A `ParticleCollection` (`opencosmo.ParticleCollection`) holds **particle
snapshots** — one `Dataset` per particle type (e.g. `dm_particles`,
`gas_particles`), with no object catalog. `oc.open` returns one when the inputs
are snapshot particle groups:

```python
import opencosmo as oc

particles = oc.open("snapshot_particles.hdf5")
# or several files, one per particle type:
particles = oc.open("dm_particles.hdf5", "gas_particles.hdf5")
```

## Keys, values, len

- Keys are **particle type names** (strings): `particles.keys()`,
  `particles["gas_particles"]` → a `Dataset` of that type's particles.
- `len(particles)` is the **number of particle types**, not particles.
  Per-type row counts via `len(particles["gas_particles"])`.

## Metadata access

```python
particles.header.parameters  # shared header; keys are dataset-dependent —
list(particles.header.parameters)  # inspect rather than assume
particles.dtype  # "snapshot_particles"
particles["gas_particles"].columns  # per-type columns
```

## Operations

The collection-level API is deliberately smaller than `Dataset`'s — **there is
no `filter`, `take`, or `sort_by` at the collection level**; apply those on the
individual datasets (`particles["gas_particles"].filter(...)`).

- `select(*columns, mode="global", **kwargs)` / `drop(...)` — applied to every
  member dataset.
- `with_datasets(names)` — restrict to a subset of particle types.
- `with_units(convention=None, conversions=None, **columns)` — applied to all
  members.
- `bound(region, select_by=None)` / `box_search(p1, p2)` — spatial selection
  applied to every member.
- `evaluate_on_dataset(func, dataset, vectorize=True, insert=True, ...)` —
  managed computation on **one named** particle dataset; returns a new
  collection or a dict when `insert=False`.
- `get_data(format="astropy", ...)` → `dict[type_name, data]`, one entry per
  particle type.

### bound() snaps to the spatial index

Because particle datasets are huge, `bound()` **snaps the query region to the
closest spatial-index cells that fully contain it** — the result may include a
small number of boundary particles outside the region. To keep exactly the
particles inside, bound the individual type afterwards (reads boundary-cell
coordinates):

```python
region = oc.make_box((100, 100, 100), (120, 120, 120))
snapped = particles.with_units("scalefree").bound(region)
exact = snapped["gas_particles"].bound(region)  # exact per-particle check
```

Regions carry no units: plain numbers are interpreted in the collection's
current unit convention, so call `with_units(...)` first if needed. Repeated
bounds compose against the logical region of the previous query.

## Performant workflow

Select columns and bound before materializing; per-type row ops go through the
child `Dataset`:

```python
region = oc.make_box((0, 0, 0), (50, 50, 50))
gas = (
    particles.with_units("comoving")
    .bound(region)["gas_particles"]  # snap, then exact-bound if needed
    .select("x", "y", "z", "mass")
    .filter(oc.col("mass") > 1e10)
)
data = gas.get_data("numpy")
```

## Limitations

- No collection-level `filter`/`take`/`sort_by` — use the per-type `Dataset`.
- Collection-level `evaluate` does not exist; use `evaluate_on_dataset`.
- `bound` is approximate (index-snapped) at the collection level.

## Common API reference

See [../api/](../api/), [../expressions.md](../expressions.md), [../mpi.md](../mpi.md).
