# Analysis: yt / pyxsim integration (`opencosmo.analysis.yt_utils`, `yt_viz`)

Tools for loading OpenCosmo particle data into `yt` and making halo visualizations.

## Optional dependencies

yt/pyxsim/matplotlib are **not** installed with the core package. Install the
"haloviz" spec with the CLI:

```bash
opencosmo install haloviz          # installs yt, pyxsim, numpy (conda-forge preferred, else pip)
```

Install `haloviz` before importing the visualization tools. Statistics are
separate; import them from `opencosmo.analysis.statistics` (see
[statistics.md](statistics.md)).

## `create_yt_dataset` — particle data → yt dataset

```python
import opencosmo as oc
from opencosmo.analysis import create_yt_dataset, ParticleProjectionPlot

with oc.open("haloproperties.hdf5", "haloparticles.hdf5").take(1, at="random") as data:
    halo = next(data.halos())  # dict: properties table + particle Datasets
    yt_ds = create_yt_dataset(halo)
    ParticleProjectionPlot(yt_ds, "z", ("dm", "particle_mass")).save()
```

Signature:
`create_yt_dataset(data, compute_xray_fields=False, return_source_model=False, pyxsim_source_model="CIESourceModel", source_model_kwargs=None, make_line_source_fields_kwargs=None)`.

- `data` is a dict of particle datasets (e.g. one object from
  `StructureCollection.objects()` / `.halos()`); only keys containing `"particles"`
  are used. Each must provide `x`, `y`, `z`, `mass` (renamed to
  `particle_position_*` / `particle_mass`); `rho`/`hh` map to `density` /
  `smoothing_length`. Bounding box is computed from the positions.
- **Unit requirement:** columns carrying `littleh` factors raise `RuntimeError` —
  convert first, e.g. `ds = ds.with_units("comoving")`. Length/mass units are set
to `Mpc` and `Msun`; cosmology and redshift are read from the dataset.
- If `("gas", "density")` is present (hydro sims), derived fields `("gas", "MMW")`,
  `temperature`, `number_density`, `xh`, `metallicity` are added automatically.
- `compute_xray_fields=True` (hydro only, else `RuntimeError`) adds pyxsim
  X-ray fields named e.g. `("gas", "xray_luminosity_0.5_2.0_keV")` /
  `xray_emissivity_...`. Defaults for `CIESourceModel`: `model="apec"`,
  `emin=0.1`, `emax=10.0` keV, `nbins=1000` (note: pyxsim downloads FITS tables on
  first use). Override via `source_model_kwargs`; other models
  (`PowerLawSourceModel`, `LineSourceModel`, ...) may need extra particle columns.
- Returns the `yt` dataset, or `(yt_ds, source_model)` when
  `return_source_model=True`.

## Plot wrappers and halo tools

- Thin wrappers with the same call signature as yt: `ProjectionPlot`, `SlicePlot`,
  `ParticleProjectionPlot`, `ProfilePlot`, `PhasePlot`.
- `visualize_halo(halo_id, data, yt_ds=None, projection_axis="z", length_scale="top left", width=None, ...)`
  → matplotlib `Figure`. `data` must be a `StructureCollection` from
  `oc.open([properties, particles])`; `halo_id` is e.g. `halo["halo_properties"]["unique_tag"]`.
  2×2 panels (DM, stars, gas, gas temperature) when dm/gravity+star+gas particles
  are all present, else a 1×N strip; raises `RuntimeError` if no supported particle
  type (`dm`/`gravity`/`star`/`gas`) is present.
- `halo_projection_array(halo_ids, data, field=("dm","particle_mass"), params=None, width=..., ...)`
  → `Figure` for a grid of halos/fields.
- `animate_halos(halo_ids, data, func="visualize_halo", rotations="y", frames=30, dpi=100, ...)`
  → `matplotlib.animation.FuncAnimation`; save with `anim.save("anim.gif", fps=10)`.

```python
from opencosmo.analysis import visualize_halo

with oc.open("haloproperties.hdf5", "haloparticles.hdf5").take(3, at="start") as data:
    for halo in data.halos():
        hid = halo["halo_properties"]["fof_halo_tag"]
        visualize_halo(hid, data).savefig(f"{hid}.png")
```

## Structure restrictions

These tools need a **StructureCollection** (properties + linked particle datasets);
a bare particle `Dataset` won't work — `get_data` is not supported on
StructureCollection, so iterate `objects()`/`halos()` and index the per-object dict.
