# take — row selection variants

## Use case
Select a small number of rows (e.g. for quick inspection or the top-N after a
sort) without materializing the full dataset. All variants are grouped here.

## Signatures (Dataset, Lightcone, StructureCollection, SimulationCollection)
```text
ds.take(n: int, at: Literal["start", "end", "random"] = "random",
        mode: Literal["local", "global"] = "local") -> SameType
ds.take_range(start: int, end: int, mode: Literal["local", "global"] = "local") -> SameType
ds.take_rows(rows: np.ndarray | DataIndex) -> SameType
```
- `take(n, at=...)`: first `n` (`"start"`), last `n` (`"end"`), or `n` random
  rows (`"random"`, the default, sampled without replacement).
- `take_range(start, end)`: standard Python range, rows `start .. end-1` (end
  exclusive).
- `take_rows(rows)`: explicit index array; any order, **no repeats**. The result
  keeps the dataset's own (sort) order, not the order of `rows`.

## Type-specific differences
- `HealpixMap.take(n, at="random")` and `take_range(start, end)` / `take_rows(rows)`
  have **no `mode`** argument; rows are pixel indices.
- `ParticleCollection`: **no take methods at all** (restricted API).

## Return shape
New immutable object of the same type with the chosen rows.

## Execution / MPI
Lazy index arithmetic. `ValueError` if `n`/indices are out of range, negative,
repeated, or `end > start`. Under MPI, `mode="local"` (default) takes `n` rows
independently per rank; `mode="global"` makes `n` the total across all ranks
(and coordinates with a global sort if the dataset is sorted). No effect without
MPI. Takes respect the current sort order — combine with
[sort_by](sort_by.md) for top-N.

## Examples
```python
import numpy as np
import opencosmo as oc

ds = oc.open("haloproperties.hdf5")
top100 = ds.sort_by("fof_halo_mass", invert=True).take(100, at="start")
sample = ds.take(100)  # 100 random rows (default at="random"); len == 100
chunk = ds.take_range(0, 1_000)  # first 1000 rows; len == 1000
some = ds.take_rows(np.array([3, 17, 42]))  # explicit rows; len == 3
```

## Pitfalls
- Default `at="random"` is **not deterministic** across calls (fresh RNG each
  time); use `at="start"`/`"end"` or `take_range` for reproducibility.
- `n > len(ds)` raises `ValueError` (it does not clamp).
- `take_rows` rejects duplicate indices.
