# Column expressions, masks, and scalar reductions

Use `oc.col(name)` expressions to keep arithmetic, dependency tracking, units,
filtering, and reductions inside OpenCosmo. Expressions are not NumPy arrays;
do not materialize columns merely to perform supported arithmetic.

## Arithmetic and units

Given a halo `ds` with the columns below:

```python
import opencosmo as oc

mass = oc.col("fof_halo_mass")
vx = oc.col("fof_halo_com_vx")
vy = oc.col("fof_halo_com_vy")
vz = oc.col("fof_halo_com_vz")
speed = (vx**2 + vy**2 + vz**2).sqrt()
query = ds.filter(mass > 1e14).select(
    "fof_halo_mass", speed=speed, kinetic_energy=0.5 * mass * speed**2
)
# Materialize only if the caller needs this table:
table = query.get_data(format="astropy", wrap_single=True)
```

Supported arithmetic includes `+`, `-`, `*`, `/`, powers, reverse arithmetic,
and methods `.sqrt()`, `.log10()`, `.exp10()`, `.arcsin()`, `.arccos()`,
`.arctan2(other)`. Do not assume arbitrary NumPy ufuncs operate on expressions.
Addition/subtraction require compatible units; inverse trig requires the
appropriate dimensionless operands. `.log10()` supports Astropy logarithmic
units; it is not permission to apply `np.log10` to a dimensional Quantity in a
custom callback. Explicitly divide by a reference unit/value when taking an
ordinary logarithm.

Public helpers live in the **singular** package:

```python
from opencosmo.column import norm_cols

speed = norm_cols("fof_halo_com_vx", "fof_halo_com_vy", "fof_halo_com_vz")
```

`add_mag_cols` and `offset_3d` are also exported there; verify their inputs for
specialized use rather than importing a nonexistent `opencosmo.columns`.

## Filtering

```python
m = oc.col("fof_halo_mass")
c = oc.col("sod_halo_cdelta")
query = ds.filter((m > 1e13) & (m < 1e14), c > 0)
# Multiple masks are ANDed; OR requires |:
alternative = ds.filter((m > 1e14) | (c > 10))
```

Use comparison operators `==`, `!=`, `<`, `<=`, `>`, `>=`, or `.isin(values)`.
Parenthesize comparisons when using `&` / `|`. Do not use Python `and` / `or`,
chained comparisons (`lower < col < upper`), or assume `~mask` is supported.

A numeric threshold uses the current column units. Put `with_units` **before**
the filter if that defines the intended threshold. Explicit compatible Astropy
quantities can make the scientific intent clearer. Filtering reads columns it
needs immediately; it does not fetch every visible column.

## Projection and registration

- `select("mass", derived=expr)` retains the specified names and derived output.
- `select("*", derived=expr)` retains **all** columns; use it only when needed.
- `with_new_columns(derived=expr)` retains existing columns and supports explicit
  descriptions/overwrite controls and array input. See its API reference.
- Register prerequisites in separate transformations when one new column needs
  another; reuse the expression itself when possible.
- Bound expressions bind to the columns available when registered. Replacing a
  visible column does not silently rewrite already registered dependencies.
  Reusable expression objects are not mutated by registration.

An expression column participates in unit conversion/inference. An externally
supplied array is a fixed value, not a cosmology-aware expression.

## Scalar-only summaries

Reducers: `.min()`, `.max()`, `.sum()`, `.mean()`, `.std()`, `.var()`,
`.median()`, `.quantile(q)` (0 <= q <= 1).

```python
m = oc.col("fof_halo_mass")
sample = ds.filter(m > 1e13)
summary = sample.select(
    min_mass=m.min(), max_mass=m.max(), mean_mass=m.mean()
).get_data()  # dict of scalar results, units retained by default
one = sample.select(mean_mass=m.mean()).get_data()  # one unwrapped scalar
```

**Do not mix scalar-only outputs with row columns** in one selection:
`ds.select("fof_halo_mass", mean_mass=m.mean())` raises `ValueError`.
Use separate selections, or use a scalar inside row-wise expression arithmetic.
Structure collections require accessing the appropriate child for scalar-only
selection. Under MPI the `mode` attached by `select`/`filter` determines reducer behavior; the default is
global. See [MPI](mpi.md).

## Managed normalization and data-dependent thresholds

```python
m = oc.col("fof_halo_mass")
sample = ds.filter(m > 1e13)
normalized = sample.select("fof_halo_mass", zscore=(m - m.mean()) / m.std())
result = normalized.get_data(format="astropy", wrap_single=True)
below_mean = sample.filter(m < m.mean())
```

Compute normalization on the intended population. Prior bounds/filters affect
it. Scalar expressions embedded in lazy columns are evaluated on the logical
rows at materialization; a scalar filter threshold is evaluated when that
filter runs. Reordering these operations can change the science.
Check nonempty samples and nonzero denominators if the population could be
constant. Do not substitute per-step statistics for a stack-wide lightcone
reduction: reductions are computed over the whole stacked lightcone.
Elementwise expressions are applied to each child.

## When expressions are not enough

Use [evaluate](api/evaluate.md) for custom numerical algorithms, preferably
column-named parameters and `vectorize=True` or bounded batches. For linked
particle calculations use structure-wise `collection.evaluate` rather than
loading every linked dataset and matching manually. Local NumPy reductions
**inside that managed callback** are appropriate. For histograms/binned
statistics, first consider [statistics](analysis/statistics.md).
