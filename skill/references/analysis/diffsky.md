# Analysis: Diffsky synthetic galaxy catalogs

OpenCosmo supports Diffsky synthetic-galaxy catalogs, including header metadata
for package versions and automatic handling of host/satellite relationships
when you filter, sort, take, or distribute a lightcone.

## 1. Installing the Diffsky packages

```bash
opencosmo install diffsky                          # diffmah, diffsky, diffstar, dsps, jax, numpy
opencosmo install diffsky --file catalog.hdf5      # pin versions from the file's header
opencosmo install diffsky --file catalog.hdf5 --dev   # development installs
```

The install command resolves package dependencies in order, preferring
conda-forge when running inside a conda environment and otherwise using pip.
Versions containing `+g<commit>` are installed from the corresponding GitHub
commit. See `opencosmo install --help` for options.

## 2. Version and catalog metadata

Diffsky-produced files have `data_type="synthetic_galaxies"` and include
metadata about the catalog and modeling packages:

```python
import opencosmo as oc

ds = oc.open("lj_487.hdf5", "lj_475.hdf5")
print(ds.header.diffsky_versions)  # package-version dictionary
print(ds.header.catalog_info)  # catalog-information dictionary
```

These header exports are dictionaries. Discover available exports via
`ds.header.parameters`. `opencosmo install diffsky --file ...` reads the
`diffsky_versions` metadata to choose package versions.

## Host and satellite relationships

OpenCosmo automatically keeps hosts and satellites consistent when selecting,
filtering, taking rows, sorting, and distributing synthetic-galaxy data. To
keep each selected galaxy's full top-host group together, enable
`keep_top_host=True` when opening:

```python
import opencosmo as oc

lc = oc.open("lj_475.hdf5", "lj_487.hdf5", keep_top_host=True)
lc = lc.filter(oc.col("logsm_obs") > 11).take(100, at="random")
```

## Typical usage

```python
import opencosmo as oc
from opencosmo.column import add_mag_cols

lc = oc.open("lj_475.hdf5", "lj_487.hdf5")
lc = lc.with_redshift_range(0, 0.1)
lc = lc.bound(oc.make_cone((40, 67), 2))
lc = lc.filter(oc.col("logsm_obs") > 10, oc.col("logsm_obs") < 12)
mag_tot = add_mag_cols("mag_g", "mag_r", "mag_i")
query = lc.select("ra", "dec", "redshift_true", total_mag=mag_tot)
data = query.get_data()  # only if needed locally
# Or write the selected query directly:
# oc.write("cut_catalog.hdf5", query)  # collective under MPI
```

Files open in the **comoving** convention by default. `.with_units("physical")`
and `.with_units("unitless")` work; `.with_units("scalefree")` raises
`ValueError` for these catalogs. Derived columns from redshift (e.g. absolute
magnitudes) work via `select`/`with_new_columns` as usual.
