# drop

## Use case
Inverse of [select](select.md): return a new object without the given columns.
Accepts the same wildcards (`"fof*"`, `"*com*"`). Useful to slim a dataset
before `get_data()` so unused large columns are never read.

## Baseline signature (Dataset, Lightcone)
```text
ds.drop(*columns: str | Iterable[str]) -> Dataset
```
No `mode` argument (dropping never needs reductions).

## Type-specific signatures
| Type | Signature notes |
|---|---|
| `StructureCollection` | `drop(*drop_args, **columns_to_drop)` — positional names drop across datasets; `dataset_name=[...]` targets a specific (possibly nested) dataset. |
| `SimulationCollection` | `drop(*args, **kwargs)` — delegates to each child. |
| `ParticleCollection` | `drop(*drop_args, **columns_to_drop)` — applies to every particle type. |
| `HealpixMap` | `drop(columns)` — takes a **single** argument (string or iterable), unlike the others. |

## Return shape
New immutable object of the same type minus the dropped columns.

## Execution
Lazy/logical; no data read.

## Examples
```python
import opencosmo as oc

ds = oc.open("haloproperties.hdf5")
ds = ds.drop("fof_halo_com_*")  # drop a wildcard group
# 'fof_halo_com_vx' is absent; 'fof_halo_mass' remains.
ds = ds.drop(["sod_halo_radius", "sod_halo_mass"])

collection = collection.drop(halo_properties=["fof_halo_tag"])
```

## Pitfalls
- `ValueError` if an exact name to drop is not in the dataset.
- `HealpixMap.drop` takes one positional argument, not `*columns`.
