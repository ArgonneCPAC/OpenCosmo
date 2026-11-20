# PowerSpectrum

A `PowerSpectrum` (`opencosmo.PowerSpectrum`) is an **eager** reader for HACC
power spectrum files: one simulation's P(k) per component at each simulation
step on one set of k bins. `oc.open` returns a `PowerSpectrum` when the file
holds a single `/power_spectrum` group.

```python
import opencosmo as oc

ps = oc.open("hydro_power_spectrum.hdf5")
```

**Eager, not MPI-distributed:** unlike datasets, a power spectrum is read fully
at `open()` time (not lazy), and it is **not** partitioned
under MPI — every rank holds the identical spectrum (under MPI only the first
rank writes it back out). There is no filter/take/bound; selection is by
component and step only.

## Metadata access (dataset-dependent)

```python
ps.components  # tuple of component names, e.g. ('agn', 'cdm', 'total', ...)
ps.steps  # simulation step numbers
ps.redshifts  # redshift of each step
ps.k  # mean wavenumber per bin, shape (n_k,), comoving
ps.n_modes  # independent modes per bin; sampling error ~ P/sqrt(n_modes)
ps.parameters  # dict of measurement parameters (the "power_spectrum"
# header block) — inspect keys, don't hard-code,
# e.g. ps.parameters.get("shot_noise_subtracted")
ps.header.parameters  # full header; ps.cosmology / ps.simulation also work
```

Gravity-only files have only the `"total"` component; hydro files add `cdm`,
`gas`, `star`, etc. Spectra are NaN at steps where a component has no particles.

## Selection and reading

Transformations return new immutable `PowerSpectrum` objects; `get_data()`
materializes (everything is already in memory, so selection is cheap):

```python
ps.select("cdm")  # subset of components
ps.drop("agn", "wind")  # exclude components
ps.with_steps(624)  # select by step number
ps.with_redshifts(0.5)  # closest step within tolerance (default 0.01;
# raises ValueError naming closest steps otherwise)
ps.with_redshifts([0, 0.5, 1], tolerance=0.05)

pks = ps.get_data()  # astropy QTable, one column per component
# plus a "redshift" column (one row per step)
pks = ps.select("cdm", "gas").get_data()  # just those components
pks = ps.get_data(format="numpy")  # dict[str, ndarray]; also pandas/polars/arrow/jax
pk = ps.select("cdm").with_steps(624).get_data()  # single component+step -> (1, n_k)
pks = ps.with_redshifts([0, 0.5]).get_data()  # subset of steps; "redshift" column kept
err = ps.select("cdm").get_data(return_error=True)  # also sampling error from n_modes
```

Shapes: `select("total").get_data()` → `(n_steps, n_k)`; a single component at
a single step keeps its leading step axis, shape `(1, n_k)` — it is **not**
unwrapped to a bare array.

## Units

Opens in the **comoving** convention (k in 1/Mpc, P in Mpc^3).
`with_units(...)` supports `"comoving"`, `"scalefree"` (h/Mpc, (Mpc/h)^3), and
`"unitless"`; requesting `"physical"` **raises an error** because power spectra
are comoving quantities.

## Limitations

- No row-level operations (no filter/take/sort/bound/evaluate); select by
  component/step/redshift only.
- Not distributed under MPI — identical on every rank; not a lazy facade.
- Optional output formats need their packages installed (same as `Dataset`).

## Common API reference

See [../api/](../api/) for `get_data` formats and [../mpi.md](../mpi.md) for MPI
notes.
