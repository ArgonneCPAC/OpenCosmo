# SimulationCollection

A `SimulationCollection` (`opencosmo.SimulationCollection`) holds **objects of
the same kind from multiple independent simulations** — a mapping of dataset
name → `Dataset` (or `Lightcone` / `StructureCollection`). Redshift steps of one
simulation belong to a `Lightcone`, not here. `oc.open` returns one when the
inputs are same-kind datasets from different simulations:

```python
import opencosmo as oc

collection = oc.open("sim_a/haloproperties.hdf5", "sim_b/haloproperties.hdf5")
```

## Keys, values, len

- Keys are the per-simulation dataset names (`collection.keys()`);
  `collection[name]` returns that simulation's child object.
- `len(collection)` is the **number of simulations** (members), not rows.
- `collection.get_data(format="astropy")` returns
  `dict[name, data]` — one materialized result per member. Only `Dataset` and
  `Lightcone` children are supported; other child kinds raise.

## Metadata access

```python
collection.header  # dict[str, OpenCosmoHeader] — one header per member
collection.dtype  # dict[str, str]
# per-member metadata via the child, e.g. collection["sim_a"].header.parameters
```

Header parameter keys are dataset-dependent: broadcast/inspect via
`{name: h.parameters for name, h in collection.header.items()}` rather than
assuming fixed names.

## Operations

Most operations **map across all children** and return a new collection
(immutable — the original is unchanged):

- `filter(*masks, **kwargs)`, `select`, `drop`, `take(n, at, mode)`,
  `take_range`, `take_rows`, `sort_by`, `bound(region)`, `with_units`,
  `with_new_columns(..., datasets=None, ...)`,
  `evaluate(func, datasets=None, insert=False, ...)`.
- `datasets=` restricts an operation to named members only.

## match / clear_match

`match(reference_name)` re-orders every member so matched objects share a row
with the reference simulation; **only rows present in every simulation
survive**, and `get_data()` then returns equal-length datasets. Requires a
pre-built matching index in the files (raises `ValueError` otherwise); only
intentionally compatible simulation suites (shared initial conditions) can be
matched.

```python
matched = collection.match("sim_a")  # sim_a order unchanged
matched = matched.filter(oc.col("fof_halo_mass") > 1e14)  # operates on sim_a,
# keeps matches elsewhere
```

**`clear_match()` does NOT undo prior operations.** It only makes *future*
operations apply uniformly to all datasets instead of the match source. The
filter above stays applied; a filter applied *after* `clear_match()` may leave
datasets with different lengths. If no match was set (or no matching index
exists), `clear_match()` has no effect.

## Limitations

- `evaluate` defaults to `insert=False` here (returns results) unlike `Dataset`.
- Matching requires the files to carry a matching index; unsupported collections
  raise instead of guessing.
- `get_data` rejects non-dataset/lightcone children (e.g. nested
  `StructureCollection` raises).

## Common API reference

See [../api/](../api/), [../expressions.md](../expressions.md), and
[../mpi.md](../mpi.md) for MPI distribution of simulation collections.
