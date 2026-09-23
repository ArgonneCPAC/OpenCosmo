"""
Convenience functions for computing various statistics on
OpenCosmo datasets.

Functions in this module should make use of OpenCosmo's builtin functionality 
for filtering and reducing data as much as possible, and should be MPI-aware. 
The calling sequence should be the same whether it is run in serial or parallel.

The companion module :py:mod:`opencosmo.analysis.plotting` wraps many of these
functions with a matplotlib front end.

Columns listed in
:py:data:`default_params <opencosmo.analysis.default_plotting_params.default_params>`
also carry a :code:`filter_bad` entry describing the sentinel values that mark
an invalid measurement. These filters are applied automatically, so the user does not
need to remember that (for example) unresolved halos are written out with
:code:`sod_halo_mass = -1`.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import opencosmo as oc
from opencosmo.analysis.default_plotting_params import default_params
from opencosmo.mpi import get_comm_world, get_mpi

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from opencosmo import Dataset, StructureCollection
    from opencosmo.column.column import Column, DerivedScalarValue

    Statistic = str | Callable[..., DerivedScalarValue]
    BinSpacing = Literal["log", "linear"]
    Mode = Literal["local", "global"]

import astropy.units as u

MPI = get_mpi()
comm = get_comm_world()
rank = comm.Get_rank() if comm is not None else 0
ranks = comm.Get_size() if comm is not None else 1


def _require_comm() -> Any:
    """
    Return the world communicator, asserting that it exists.

    The MPI branches in this module are guarded by :code:`ranks > 1`, which can
    only be true when a communicator is present. That invariant is invisible to
    a type checker, so those branches route through this helper instead of
    narrowing :code:`comm` at each call site.
    """
    assert comm is not None
    return comm


def _get_statistic(
    col: Column, statistic: Statistic, **kwargs: Any
) -> DerivedScalarValue:
    """
    Build the scalar reduction described by ``statistic`` for a given column.

    Returns either the result of a custom, callable input function or that of
    a builtin method attached to the Column object. Note that nothing is
    computed here: the returned object is a lazy
    :code:`DerivedScalarValue` that is only evaluated when it is passed into
    :py:meth:`select <opencosmo.Dataset.select>` and the data is read.

    Parameters
    ----------
    col : opencosmo.column.Column
        The column to reduce, e.g. :code:`oc.col("sod_halo_cdelta")`.
    statistic : str or Callable
        The name of a builtin reduction on the column ("mean", "std", "var",
        "min", "max", "median", "sum", "quantile"), the same name prefixed
        with "geometric_", or a callable taking the column and returning a
        scalar reduction of it.
    **kwargs
        Forwarded to the reduction, e.g. :code:`q=0.84` for "quantile".

    Returns
    -------
    statistic : opencosmo.column.DerivedScalarValue
        The unevaluated reduction.

    Raises
    ------
    TypeError
        If ``statistic`` is neither a string nor callable.
    """

    if isinstance(statistic, str):
        if statistic.startswith("geometric_"):
            # if doing geometric mean, median, quantile, etc., does
            # global scalar reduction in log space. Will need to
            # convert back to exp10() space after the function call.

            return getattr( col.log10(), statistic.removeprefix("geometric_") )(**kwargs)

        else:
            return getattr(col, statistic)(**kwargs)

    if callable(statistic):
        return statistic(col, **kwargs)

    raise TypeError("statistic must be a string or callable")

def _compute_bins(
    min: Any, max: Any, n: int, bin_spacing: BinSpacing = "log"
) -> np.ndarray:
    """
    Construct ``n`` bin edges spanning ``[min, max]``.

    Parameters
    ----------
    min, max : float or astropy.units.Quantity
        The endpoints of the binning range, inclusive. If these carry units,
        the edges are computed on the bare values and the unit is reattached
        to the result, so the returned edges carry the unit of ``min``.
    n : int
        The number of bin *edges* to produce. This is one more than the number
        of bins.
    bin_spacing : str, "log" or "linear", default = "log"
        Whether the edges should be spaced geometrically or arithmetically.

    Returns
    -------
    edges : numpy.ndarray or astropy.units.Quantity
        The ``n`` bin edges, carrying the unit of ``min`` if it had one.

    Raises
    ------
    RuntimeError
        If ``bin_spacing`` is not "log" or "linear".
    """

    if isinstance(min, u.Quantity):
        # note that this function is only called if bin edges are not explicitly 
        # passed in by the user, in which case the units for min and max 
        # should be guaranteed to be in the same units. 
        # Convert units of max to those of min again just to be safe.
        unit = min.unit

        min = min.value
        max = max.to(unit).value
    else:
        unit = None


    if bin_spacing == "log":
        bins = np.geomspace(min, max, n)
    elif bin_spacing == "linear":
        bins = np.linspace(min, max, n)
    else:
        raise RuntimeError(f"unrecognized value for bin_spacing: {bin_spacing}")

    if unit is not None:
        bins = bins * unit

    return bins

def _filter_bad(ds: Dataset, column: str) -> Dataset:
    """
    Drop rows where ``column`` holds a sentinel or unphysical value.

    The masks are read from the ``filter_bad`` entry of
    :py:data:`default_params <opencosmo.analysis.default_plotting_params.default_params>`.
    Columns that have no entry, or whose entry sets ``filter_bad`` to
    :code:`None`, are returned unchanged, so this is always safe to call.

    Parameters
    ----------
    ds : opencosmo.Dataset
        The dataset to filter.
    column : str
        The column whose default filters should be applied.

    Returns
    -------
    ds : opencosmo.Dataset
        The filtered dataset, or the original dataset if no filters apply.
    """

    params = default_params.get(column)
    if params is None:
        return ds

    filters = params.get("filter_bad")
    if filters is None:
        return ds

    if not isinstance(filters, (list, tuple)):
        filters = [filters]

    return ds.filter(*filters)


def binned_statistic(
    ds: Dataset | StructureCollection,
    column: str,
    bin_by: str = "sod_halo_mass",
    statistic: Statistic = "mean",
    bins: int | Sequence = 20,
    bin_spacing: BinSpacing = "linear",
    dataset: str = "halo_properties",
    mode: Mode = "global",
    **kwargs: Any,
) -> tuple[list, np.ndarray | Sequence]:
    r"""
    Compute a statistic of ``column`` in bins of ``bin_by``.

    The statistic can be any of the prebuilt scalar reductions (in string form --
    e.g. "mean", "std"), a geometric variant of one, or a custom statistic that
    takes the column as input.

    Nothing is read from disk until each bin's reduction is evaluated, and
    only one scalar per bin is ever materialized, so this is safe to run on
    datasets far larger than memory.

    This function is MPI-aware. When ``mode = "global"``, the scalar reduction
    for each bin is performed across all ranks. When ``mode = "local"``, the reduction
    is performed on each rank separately.

    .. code-block:: python

        import opencosmo as oc
        from opencosmo.analysis.statistics import binned_statistic

        ds = oc.open("haloproperties.hdf5")

        # mean concentration in 20 log-spaced bins of M200c
        c, edges = binned_statistic(ds, "sod_halo_cdelta", bin_spacing="log", statistic="mean")

        # log-space scatter in Y500c
        scatter, edges = binned_statistic(
            ds, "sod_halo_Y500c", bin_spacing="log", statistic="geometric_std"
        )

        # a custom reduction
        def dynamic_range(col):
            return (col.max() - col.min()) / col.std()

        dr, edges = binned_statistic(ds, "sod_halo_cdelta", statistic=dynamic_range)

    Parameters
    ----------
    ds : opencosmo.Dataset or opencosmo.StructureCollection
    column : str
        The column to compute the statistic of.
    bin_by : str, default = "sod_halo_mass"
        The column whose values define the bins.
    statistic : str or Callable, default = "mean"
        The reduction to apply within each bin. Accepts the name of any scalar
        reduction defined on a column -- "mean", "std", "var", "min", "max",
        "median", "sum", or "quantile" -- optionally prefixed with
        "geometric\_", in which case the reduction is performed on
        :math:`\log_{10}` of the column and the result is raised back through
        :math:`10^x`. For example, "geometric_mean" gives the mean in log space, and
        "geometric_std" gives the lognormal scatter.
        A custom, callable function is also accepted; it receives the column and must return a
        scalar reduction of it (see the example above).
    bins : int or list, default = 20
        The number of bins, or an explicit list of bin edges. When an integer
        is given, ``bins + 1`` edges are placed between the global
        minimum and maximum of ``bin_by``.
    bin_spacing : str, "log" or "linear", default = "linear"
        How automatically generated edges are spaced. Ignored when explicit
        edges are given via ``bins``.
    dataset : str, default = "halo_properties"
        Which member dataset to use when ``ds`` is a collection. Ignored
        otherwise.
    mode : str, "local" or "global", default = "global"
        How scalar reductions are combined under MPI. ``"global"`` reduces
        across all ranks, so every rank computes the same bin edges and the
        same per-bin statistic over the full dataset. ``"local"`` gives each
        rank the statistic of its own rows. Has no effect when not running
        under MPI.
    **kwargs
        Forwarded to the statistic, e.g. :code:`q=0.84` when
        :code:`statistic="quantile"`.

    Returns
    -------
    binned_stat : list
        The value of the statistic in each bin, of length ``len(bins) - 1``.
    bins : numpy.ndarray or list
        The bin edges, of length ``len(binned_stat) + 1``. These are the edges
        that were passed in, or the ones that were computed from the data.

    Raises
    ------
    TypeError
        If ``statistic`` is neither a string nor callable.
    """

    source = ds[dataset] if isinstance(ds, oc.StructureCollection) else ds
    assert not isinstance(source, oc.StructureCollection)

    source = _filter_bad(source, column)

    if isinstance(bins, int):

        d = source.select(
                bin_min = oc.col(bin_by).min(), 
                bin_max = oc.col(bin_by).max(),
                mode = mode,
            ).get_data()

        edges: np.ndarray | Sequence = _compute_bins(
            d["bin_min"], d["bin_max"], bins+1, 
            bin_spacing = bin_spacing
        )

    else:
        edges = bins

    binned_stat: list = []

    for i in range(len(edges)-1):
        low, high = edges[i], edges[i+1]

        # get_data() is typed as a union over table/array/dict, but selecting a
        # single scalar reduction always unpacks to a scalar here.
        stat_value: Any = (
            source.filter(oc.col(bin_by) >= low, oc.col(bin_by) < high)
            .select( 
                stat = _get_statistic(oc.col(column), statistic, **kwargs),
                mode = mode,
            )
            .get_data()
        )

        # convert back from log-space if doing geometric mean, median, etc.
        if isinstance(statistic, str) and statistic.startswith("geometric_"):
            stat_value = 10 ** stat_value

        binned_stat.append(stat_value)

        if ranks > 1:
            _require_comm().Barrier()

    return binned_stat, edges

def hist1d(
    ds: Dataset | StructureCollection,
    column: str,
    bins: int | Sequence = 20,
    bin_spacing: BinSpacing = "linear",
    dataset: str = "halo_properties",
    mode: Mode = "global",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a one-dimensional histogram of ``column``.

    Unlike :py:func:`binned_statistic`, which performs one lazy reduction per
    bin, this function reads ``column`` into memory on each rank and hands it
    to :py:func:`numpy.histogram`. Under MPI (with `mode = "global"`) the per-rank counts are then
    summed with an :code:`allreduce`, so every rank receives the histogram of
    the full dataset.

    .. code-block:: python

        import opencosmo as oc
        from opencosmo.analysis.statistics import hist1d

        ds = oc.open("haloproperties.hdf5")
        counts, edges = hist1d(ds, "sod_halo_mass", bins=30, bin_spacing="log")

    Parameters
    ----------
    ds : opencosmo.Dataset or opencosmo.StructureCollection
    column : str
        The column to histogram. e.g., "sod_halo_mass"
    bins : int or sequence, default = 20
        The number of bins, or an explicit sequence of bin edges. When an
        integer is given, ``bins + 1`` edges are placed between the global
        minimum and maximum of ``column`` according to ``bin_spacing``.
    bin_spacing : str, "log" or "linear", default = "linear"
        How automatically generated bin edges are spaced. Ignored when explicit
        edges are given.
    dataset : str, default = "halo_properties"
        Which member dataset to use when ``ds`` is a collection. Ignored
        otherwise.
    mode : str, "local" or "global", default = "global"
        How the histogram is computed under MPI. ``"global"`` reduces
        across all ranks, so every rank computes the same bin edges and the
        same bin counts are summed across all ranks. ``"local"`` computes 
        a separate histogram on each rank. Has no effect when not running
        under MPI.

    Returns
    -------
    counts : numpy.ndarray
        The number of entries in each bin, of length ``len(bin_edges) - 1``.
        Summed over all ranks under MPI.
    bin_edges : numpy.ndarray
        The bin edges, of length ``len(counts) + 1``. These carry units when
        they were derived from a unit-ful column.

    Raises
    ------
    RuntimeError
        If ``bin_spacing`` is not "log" or "linear".
    """

    source = ds[dataset] if isinstance(ds, oc.StructureCollection) else ds
    assert not isinstance(source, oc.StructureCollection)

    source = _filter_bad(source, column)

    if isinstance(bins, int):

        d = source.select(
                bin_min = oc.col(column).min(), 
                bin_max = oc.col(column).max(),
                mode = mode,
            ).get_data()

        edges: np.ndarray | Sequence = _compute_bins(
            d["bin_min"], d["bin_max"], bins+1, 
            bin_spacing = bin_spacing
        )

    else:
        edges = bins

    counts, bin_edges = np.histogram(
        source.select(column).get_data(), bins=edges
    )

    if ranks > 1 and mode == "global":
        counts = _require_comm().allreduce( counts, op=MPI.SUM )

    return counts, bin_edges

def hist2d(
    ds: Dataset | StructureCollection,
    column_x: str,
    column_y: str,
    bins: int | Sequence = 100,
    bin_spacing: BinSpacing | Sequence[BinSpacing] = "linear",
    dataset: str = "halo_properties",
    mode: Mode = "global",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a two-dimensional histogram of ``column_y`` against ``column_x``.

    Both columns are read into memory on each rank and passed to
    :py:func:`numpy.histogram2d`. Under MPI (with `mode="global"`) the per-rank count grids are summed
    with an :code:`allreduce`, so every rank receives the histogram of the full
    dataset.

    .. code-block:: python

        import opencosmo as oc
        from opencosmo.analysis.statistics import hist2d

        ds = oc.open("haloproperties.hdf5")

        # concentration against mass: log bins in mass, linear in concentration
        h, x, y = hist2d(
            ds,
            "sod_halo_mass",
            "sod_halo_cdelta",
            bin_spacing=("log", "linear"),
        )

    Parameters
    ----------
    ds : opencosmo.Dataset or opencosmo.StructureCollection
    column_x, column_y : str
        The columns to histogram along the first and second axes.
    bins : int or sequence, default = 100
        The number of bins along *each* axis, or a two-element sequence
        ``[bins_x, bins_y]`` of explicit edge arrays. When an integer is given,
        ``bins + 1`` edges are placed between the global minimum and maximum of
        each column independently.
    bin_spacing : str or tuple of str, default = "linear"
        How automatically generated edges are spaced. A single string applies
        to both axes; a two-element sequence ``(spacing_x, spacing_y)`` sets
        them independently. Each entry must be "log" or "linear". Ignored when
        explicit edges are given.
    dataset : str, default = "halo_properties"
        Which member dataset to use when ``ds`` is a collection. Ignored
        otherwise.
    mode : str, "local" or "global", default = "global"
        How the 2d histogram is computed under MPI. ``"global"`` reduces across all ranks
        to create one integrated 2d histogram, while ``"local"`` computes a separate 
        2d histogram on each rank.

    Returns
    -------
    h : numpy.ndarray
        The counts, with shape ``(len(x) - 1, len(y) - 1)``. Following the
        convention of :py:func:`numpy.histogram2d`, the first axis indexes
        ``column_x``, so ``h`` must be transposed before being handed to
        :py:func:`matplotlib.pyplot.pcolormesh`. Summed over all ranks under
        MPI with `mode = "global"`.
    x : numpy.ndarray
        The bin edges along ``column_x``. These carry units when they were
        derived from a unit-ful column.
    y : numpy.ndarray
        The bin edges along ``column_y``. These carry units when they were
        derived from a unit-ful column.

    Raises
    ------
    RuntimeError
        If ``bin_spacing`` is neither a string nor a sequence, or if an entry
        is not "log" or "linear".
    """

    source = ds[dataset] if isinstance(ds, oc.StructureCollection) else ds
    assert not isinstance(source, oc.StructureCollection)

    source = _filter_bad(source, column_x)
    source = _filter_bad(source, column_y)

    if isinstance(bins, int):
        d_x = source.select(
                bin_min = oc.col(column_x).min(), 
                bin_max = oc.col(column_x).max(),
                mode = mode,
            ).get_data()

        d_y = source.select(
                bin_min = oc.col(column_y).min(), 
                bin_max = oc.col(column_y).max(),
                mode = mode,
            ).get_data()

        if isinstance(bin_spacing, (list, tuple)):
            bin_spacing_x, bin_spacing_y = bin_spacing
        elif isinstance(bin_spacing, str):
            bin_spacing_x = bin_spacing
            bin_spacing_y = bin_spacing
        else:
            raise RuntimeError(f"Invalid input for bin_spacing: {bin_spacing}")
    
        bins_x = _compute_bins(
            d_x["bin_min"], d_x["bin_max"], bins+1, 
            bin_spacing = bin_spacing_x
        )

        bins_y = _compute_bins(
            d_y["bin_min"], d_y["bin_max"], bins+1, 
            bin_spacing = bin_spacing_y
        )

    else:
        bins_x, bins_y = bins

    data = source.select(column_x, column_y).get_data()

    h, x, y = np.histogram2d(data[column_x], data[column_y], bins = [bins_x, bins_y])

    if ranks > 1 and mode == "global":
        h = _require_comm().allreduce( h, op=MPI.SUM )

    return h, x, y