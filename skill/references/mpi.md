# MPI execution

Run the same analysis script on every rank, e.g. `mpiexec -n 2 python script.py`.
`oc.open` partitions supported datasets when an MPI communicator is available.
MPI support requires `mpi4py` and a working MPI runtime; parallel writes need
MPI-enabled h5py. Reading does not require an MPI-enabled h5py build.

```python
from opencosmo.mpi import get_comm_world

comm = get_comm_world()  # None if MPI is unavailable or world size is one
rank = 0 if comm is None else comm.Get_rank()
```

## All ranks enter collectives

All participating ranks must execute opening, writing, global reductions,
`analysis.gather`/`reduce`, and global statistics in the same order. Arguments
must agree where the operation requires it (paths, column names, bins, etc.).
Never put a collective in `if rank == 0`. Only processing an already-computed
root-owned result belongs there. Empty ranks must still participate.

This is WRONG: `if rank == 0: total = comm.allreduce(local_count)`.
Compute `total = comm.allreduce(local_count)` on **all** ranks, then print on root.

OpenCosmo's optional `abort_on_error` helper can abort a script after an
exception rather than leaving peer ranks waiting.

## Local/global semantics

| Operation | Default behavior |
|---|---|
| `filter`, `select`, `with_new_columns` | `mode="global"` attaches cross-rank reducers; ordinary elementwise work remains local |
| `take`, `take_range` | `mode="local"`: count/range per rank; explicit `"global"` uses the distributed row space |
| `take_rows` | No `mode`; supplied positions are local |
| Standard `get_data` | Local selected rows; selected global scalars are globally reduced |
| `analysis.gather`, `analysis.reduce` | Results on root by default; `all=True` distributes results to all ranks |

Check type-specific signatures: not every facade accepts `mode` (maps do not
provide distributed catalog semantics).

For global scalar reductions prefer expressions to gathering a whole table.
MPI `min`/`max`/`sum`/`mean` use collective reductions. `var`/`std`/`median`/
`quantile` currently gather values to root and broadcast the result: budget
root memory. Binned statistics still invoke their requested reducer; requesting
variance/quantiles there does not magically remove this cost.

## Opening distribution

- `mpi_mode="spatial"` is the default: ranks see the files and partition rows.
- `mpi_mode="redshift"` assigns contiguous redshift-step groups weighted by
  approximate data volume (top-level row counts), not equal numbers of steps.
  Linked files for a supported structure step are assigned together.
- Unsupported layouts, including nested subtype lightcones, fall back to
  spatial distribution. A snapshot with redshift mode also uses fallback.
- Surplus ranks open an empty-schema reference and still participate. Do not
  assume all ranks have the same child keys in redshift mode.

Use redshift mode when the file/layout organization supports it; do not rely
on it to assign individual objects or produce globally sorted output.

## Executable summary pattern

Set `path` to a real halo-properties file. Every rank runs the entire script:

```python
import opencosmo as oc
from opencosmo.mpi import get_comm_world

comm = get_comm_world()
rank = 0 if comm is None else comm.Get_rank()
with oc.open(path) as ds:
    sample = ds.filter(oc.col("fof_halo_mass") > 1e14)
    m = oc.col("fof_halo_mass")
    summary = sample.select(total_mass=m.sum(), mean_mass=m.mean()).get_data()
    # Count does not require materializing another column.
    count = len(sample) if comm is None else comm.allreduce(len(sample))
    if rank == 0:
        print(count, summary)
```

For a custom callback, use [analysis.reduce](analysis/distributed.md) after
selection; for a global final table, use `gather`. Do not call local `get_data`
then reconstruct a global result manually unless no supported operation fits.

## Writing and plotting

Call `oc.write(path, query)` on **every rank**, using the same output path.
Do not call `get_data` first. Parallel writes need MPI-enabled h5py; otherwise
OpenCosmo writes ranks sequentially. All-empty writes raise `ValueError`. See [write](api/write.md).

Compute/gather globally on all ranks, then plot or save root-owned results.
The plotting wrappers also compute collectively before drawing on root; do
not wrap the whole plotting call in a root-only branch. `all=False` can return
`(None, None)` on non-root ranks. Check ownership before figure operations.
