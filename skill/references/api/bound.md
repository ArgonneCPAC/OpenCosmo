# bound — spatial queries

## Use case
Restrict a dataset/collection to a spatial region. Build regions with the
public helpers. 3D box coordinates are unitless and interact with the object's
unit convention; sky coordinates and angular radii accept Astropy quantities
and default to degrees when given as plain numbers.

```python
oc.make_box(p1, p2)  # 3D box, two corner points (3-tuples)
oc.make_cone(
    center, radius
)  # sky cone; center (ra, dec) in deg or SkyCoord, radius deg or Quantity
oc.make_skybox(
    p1, p2
)  # 2D box on the sky (small regions only); p2 may be a size -> center/size form
```

## Baseline signature (Dataset, Lightcone, StructureCollection, SimulationCollection)
```text
ds.bound(region: Region, select_by: str | None = None) -> SameType
```
- **Exact**: coordinates of objects in boundary cells are read and checked, so
  only rows truly inside the region are kept. No `snap` argument exists.
- `select_by`: which coordinate set to query with when a dataset has several.
- Raises `AttributeError` if the dataset has no spatial index.
- Sky objects (lightcones): use `make_cone`/`make_skybox`; also available:
  `cone_search(center, radius)`, `box_search(p1, p2)`, `pixel_search(pixels, nside=64)`,
  `get_pixels(nside=64)`, and `with_redshift_range(z_low, z_high)`.

## Type-specific behavior
| Type | Notes |
|---|---|
| `ParticleCollection` | `bound(region, select_by=None)` **snaps to whole spatial-index cells** — approximate; boundary cells may include particles outside the region. For exact cuts: `collection["gas_particles"].bound(region)` (a `Dataset.bound`). Also `box_search(p1, p2)` = `bound(make_box(...))`. Repeated bounds intersect logically with the previous region. |
| `HealpixMap` | `bound(region, inclusive: bool = False)` selects pixel centers by default; `inclusive=True` also keeps overlapping pixels. `cone_search` currently uses the default (center-inclusion), despite an inconsistent docstring. Also supports `box_search`/`cutouts`. |
| `StructureCollection` | `bound(region, select_by=None)` — bounds the properties dataset; linked datasets are re-linked to the surviving rows. |

## Return shape
New immutable object of the same type restricted to the region.

## Execution / units / MPI
Uses spatial-index cells (nested HEALPix for lightcones):
fully contained chunks are taken without reading coordinates; only boundary
chunks read coordinates. Region values are converted to the current convention
— call [with_units](with_units.md) first to query in another convention. Under
MPI, spatial partitioning is handled at open time; each rank queries its own
rows.

## Examples
```python
import opencosmo as oc

ds = oc.open("haloproperties.hdf5")
box = oc.make_box((100, 100, 100), (120, 120, 120))
ds_sub = ds.bound(box)
print(len(ds_sub))  # number of objects inside the box

# structure collection: bounds the properties dataset, re-links particles
coll = oc.open("haloproperties.hdf5", "haloparticles.hdf5")
bounded = coll.bound(box)
print(len(bounded), len(bounded["halo_properties"]))  # 2 2
print(len(bounded["gas_particles"]))  # linked rows for the selected structures

# Lightcone assembled from explicit redshift-step files.
steps = ["step_600/haloproperties.hdf5", "step_601/haloproperties.hdf5"]
lc = oc.open(*steps)
cone = oc.make_cone((150.0, 2.0), 1.0)  # ra, dec in deg; radius in deg
print(len(lc.bound(cone)))  # number of rows inside the cone
lc_sub = lc.bound(cone).with_redshift_range(0.5, 1.0)

# particles: bound the collection, then index the linked particle dataset
# (bounding a linked particle dataset directly needs resolvable coordinate
# columns; the collection routes the bound through halo centers and re-links)
arb = oc.open("haloproperties.hdf5", "haloparticles.hdf5")
gas_of_selected_halos = arb.bound(oc.make_box((100,) * 3, (120,) * 3))[
    "gas_particles"
]  # linked particles are NOT individually bounded
# on a standalone ParticleCollection (snapshot_particles file), bound() is
# cell-snapped (approximate); for exact cuts use particle_collection[name].bound(...)
```

## Pitfalls
- `make_box` with a zero-width dimension raises `ValueError`.
- `make_skybox` is only truly rectangular for small sky areas.
- `ParticleCollection.bound` is approximate (cell-snapped); do not use it when
  exact boundaries matter.
- Querying a region not contained in the dataset's current region warns and may
  return empty results.
