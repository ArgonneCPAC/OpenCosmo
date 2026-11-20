# sort_by

## Use case
Order rows by a column — typically to take the largest/smallest objects
(`sort_by(...).take(n, at="start")`) — without reading data yet.

## Baseline signature (Dataset, Lightcone, SimulationCollection)
```text
ds.sort_by(column: str | None, invert: bool = False) -> SameType
```
- Ascending by default; `invert=True` for descending. `column=None` removes
  sorting. `ValueError` if the column does not exist.
- `StructureCollection.sort_by(column, invert=False)` (no `None`).
- `HealpixMap.sort_by(column, invert=False)` — mostly for pixel ordering.
- `ParticleCollection`: **unsupported**.

## Return shape
New immutable object of the same type with a sort key recorded. The current key
is exposed via the `sorted_by` property (`None` if unsorted).

## Execution / MPI
- `sort_by` itself is **purely logical** — it only records the sort key in the
  lazily; no column data is read until needed for sorting.
- The sort column **is read when a `take*` is applied to a sorted dataset**
  (`get_sorted_index` remaps the taken rows through the global ordering), and
  again at materialization (`__sort_data`) to order the output columns.
  So `sort_by(...).take(n, at="start")` reads the sort column at `take` time.
- Under MPI, `take(..., mode="global")` and `sort_by` coordinate so ranks take
  from the globally sorted dataset.

## Examples
```python
import opencosmo as oc

ds = oc.open("haloproperties.hdf5")
biggest = ds.sort_by("fof_halo_mass", invert=True).take(5, at="start")
masses = biggest.select("fof_halo_mass").get_data()  # bare Quantity (single column)
# masses == [9.8098207e+14, 9.4394087e+14, 4.6863240e+14, 4.5335883e+14, 4.1522834e+14] solMass
print(biggest.sorted_by)  # 'fof_halo_mass'

unsorted = biggest.sort_by(None)  # remove sorting (logical only)
```

## Pitfalls
- Sorting is not free: the sort column is read for the whole dataset (at
  `take*` time if combined with a take, and at materialization). Filter
  first, then sort the survivors.
- `sorted_by` returns only the column name, not the direction.
