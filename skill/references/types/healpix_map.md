# HealpixMap

A `HealpixMap` (`opencosmo.HealpixMap`) wraps **HEALPix map data** (one layer)
plus explicit coverage metadata. `oc.open` returns one when the file contains
a healpix map group (a `/data` group whose columns are pixel-indexed map values).

```python
import opencosmo as oc

m = oc.open("map.hdf5")
```

## Coverage and metadata

Coverage is explicit metadata on the map — query it, don't assume full sky:

```python
m.nside  # high-resolution nside of the map data
m.nside_lr  # low-resolution nside used for healsparse access
m.ordering  # "NESTED" (maps are always stored NESTED)
m.full_sky  # bool — full coverage or a partial map
m.pixels  # HEALPix pixel indices; can allocate a large array
m.z_range  # redshift range the map covers
m.region  # sky region covered
m.columns  # map column names (dataset-dependent — use m.columns, not literals)
m.descriptions
m.header.parameters
list(m.header.parameters)  # header blocks; keys dataset-dependent
len(m)  # number of pixels in the map (nside2npix(nside) when full sky)
```

`full_sky` reflects whether the current map contains all sky pixels. Row
selections can make it false; a sort alone does not discard pixels. Column
selections preserve coverage. `pixels` and `region` can allocate arrays for all
selected pixels, even without reading map values; on large full-sky maps avoid
requesting them merely to inspect the schema.

## Reading data

```python
m.get_data()  # healsparse format by default (needs healsparse):
# dict[str, HealSparseMap], one per column
m.get_data("healpix")  # dict[str, ndarray] of nested-order HEALPix
# arrays (partial sky uses UNSEEN for uncovered pixels)
m.get_data("raw")  # (pixels, dict[str, ndarray]) tuple
m.get_data(nside_out=64)  # downsampled to another nside
```

Like `Dataset.get_data`, a map with a **single remaining column is unwrapped**:
`m.select("kappa").get_data()` returns a bare `HealSparseMap` and
`m.select("kappa").get_data("healpix")` a bare `ndarray`.

## Operations

- Spatial: `bound(region, inclusive=False)` (pixel **centers** in region by
  default; `inclusive=True` also keeps overlapping pixels),
  `cone_search(center, radius)` delegates to `bound(make_cone(...))` using
  center-inclusion, despite an inconsistent `cone_search` docstring. To include
  overlapping pixels explicitly use `bound(oc.make_cone(...), inclusive=True)`;
  `box_search(p1, p2)` and `cutouts(centers, angular_size, npix=64)` are also available.
- `with_resolution(nside)` — downsample to a **lower** nside (values averaged
  per low-res pixel). Passing the **same nside is a no-op** (returns the same
  result); a **higher nside raises `ValueError`**.
- Row/column ops mirror datasets: `filter`, `select`/`drop`,
  `with_new_columns`, `take`/`take_range`/`take_rows`, `sort_by`, `rows()`,
  `evaluate(func, format="numpy", ...)`.

```python
import astropy.units as u

m = m.bound(oc.make_cone((150.0, 2.0), 5 * u.deg))  # cone query in degrees
m = m.select("ksz", "tsz")  # map columns from m.columns
sm = m.with_resolution(64)  # downsample; same nside = no-op
data = m.get_data("healsparse")  # dict[str, HealSparseMap]
```

## Limitations

- **`with_units` is NOT supported** — calling it raises `NotImplementedError`:
  maps are redshift-integrated observed quantities, so unit conversion is
  non-trivial.
- `"healpix"` can represent partial coverage, but allocates a full-sky array
  with uncovered pixels masked/UNSEEN. Prefer `"raw"` or `"healsparse"` for
  small regions and large `nside` to avoid this allocation. Current `bound`
  may itself expand pixel indices, so bounding is not a guarantee of constant
  memory on a very high-resolution full-sky map.
- `with_resolution` cannot increase resolution.
- Not a mapping over rows in the usual sense: `len(m)` is the pixel count, and
  `m[name]` is not column access — use `select` + `get_data`.

## Common API reference

See [../api/](../api/), [../expressions.md](../expressions.md), [../mpi.md](../mpi.md).
