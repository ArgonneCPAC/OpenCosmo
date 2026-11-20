# write — persist datasets/collections

## Use case
Write a `Dataset` or collection to an OpenCosmo HDF5 file.
The dataset stays lazy until write time; write after transformations so only
the final selection is persisted.

## Signatures
```text
oc.write(path: Path, dataset: Writeable, overwrite: bool = False,
         compress: bool = True, **kwargs) -> None
oc.io.write_parquet(path, to_write, overwrite: bool = False, **kwargs)  # requires pyarrow
```
(`write_parquet` is exposed lazily; importing it without PyArrow raises
`ImportError`.)

## Behavior
- `oc.write` validates the path: `FileExistsError` if it exists and
  `overwrite=False`; `FileNotFoundError` if the parent directory is missing.
- Output structure is determined from the object. Under MPI, all ranks must
  call `write` together.
- `compress=True` applies HDF5 compression where applicable.
- Works for `Dataset`, `Lightcone`, `StructureCollection`,
  `SimulationCollection`, `HealpixMap`, and `ParticleCollection`.

## Examples
```python
import astropy.units as u
import opencosmo as oc

ds = oc.open("haloproperties.hdf5")
ds = ds.filter(oc.col("fof_halo_mass") > 1e14 * u.Msun).select(
    "fof_halo_mass", "fof_halo_center_*"
)
oc.write("massive_halos.hdf5", ds)  # FileExistsError if it exists
oc.write("massive_halos.hdf5", ds, overwrite=True)

# parquet export of a dataset (needs pyarrow)
oc.io.write_parquet("massive_halos.parquet", ds)
```

## Pitfalls
- Writing an already-open output path without `overwrite=True` fails fast with
  `FileExistsError` — intended, to avoid clobbering inputs.
- Under MPI all ranks must call `write` collectively.
