# with_units

## Use case
Switch the unit convention of a dataset and/or convert individual units (e.g.
Mpc → light-year). Files open in the **comoving** convention by default.
Conventions: `"comoving"` (default), `"physical"`, `"scalefree"`, `"unitless"`.

## Baseline signature (Dataset, Lightcone, SimulationCollection)
```text
ds.with_units(convention: Literal["scalefree", "comoving", "physical", "unitless"] | None = None,
              conversions: dict[u.UnitBase, u.UnitBase] | None = None,
              **columns: u.UnitBase) -> SameType
```
```python
import astropy.units as u

ds = ds.with_units("physical", fof_halo_mass=u.kg)  # convention + one column
ds = ds.with_units(
    conversions={u.Mpc: u.lyr}, fof_halo_center_x=u.km
)  # blanket + per-column
ds = ds.with_units()  # clears existing conversions
```

## Semantics
- Conversions are **lazy**: applied when data is materialized, not at call time.
- Order: convention change first, then blanket `conversions`, then per-column
  `**columns` (highest precedence). Changing convention **clears** previous
  conversions. Calling with no arguments clears conversions.
- `"scalefree"` mass units are `solMass / littleh`, which cannot convert to
  `kg` (a per-column `fof_halo_mass=u.kg` under `"scalefree"` raises
  `ValueError: Cannot convert values with units solMass / littleh to value
  with units kg` at the `with_units` call). In `"physical"` the same column
  is plain `solMass` and converts to `kg`.

## Type-specific behavior
| Type | Notes |
|---|---|
| `StructureCollection` | `with_units(convention=None, conversions=None, **dataset_conversions)` — per-dataset settings via `dataset_name={...}` kwargs (dot paths for nested). |
| `ParticleCollection` | Same signature as Dataset, applied to every particle type; a column conversion applies to every type having that column. `ValueError` if no type has a named column. |
| `HealpixMap` | **Raises `NotImplementedError`** — maps are redshift-integrated observed quantities, so unit conversion is not supported. |

## Return shape
New immutable object of the same type with the new unit settings.

## Execution / MPI
This is a lazy transformation; conversions are applied when data is
materialized. Incompatible per-column conversions fail fast with `ValueError`
at the `with_units` call. The same settings apply on every MPI rank.

## Examples
```python
import astropy.units as u
import opencosmo as oc

ds = oc.open("haloproperties.hdf5")  # comoving convention
ds_phys = ds.with_units("physical")
# The materialized mass column is expressed in solMass.
ds_lyr = ds.with_units(
    fof_halo_center_x=u.lyr, fof_halo_center_y=u.lyr, fof_halo_center_z=u.lyr
)
table = ds_lyr.get_data()  # columns now in requested units
```

## Pitfalls
- Derived columns created with `oc.col` are unit-aware and transform correctly;
  columns added as raw arrays/`Quantity` or via `evaluate` do **not** change
  under `with_units`.
- `HealpixMap.with_units(...)` always raises `NotImplementedError`.
- A convention switch silently drops earlier per-column conversions.
