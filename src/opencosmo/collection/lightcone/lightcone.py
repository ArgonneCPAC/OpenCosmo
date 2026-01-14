from __future__ import annotations

from collections import OrderedDict, defaultdict
from functools import cached_property, reduce
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    Mapping,
    Optional,
    Self,
    Sequence,
)

import numpy as np
from astropy.table import vstack  # type: ignore

import opencosmo as oc
from opencosmo.collection.lightcone.stack import stack_lightcone_datasets_in_schema
from opencosmo.column.column import DerivedColumn
from opencosmo.dataset import Dataset
from opencosmo.dataset.formats import convert_data, verify_format
from opencosmo.evaluate import prepare_kwargs
from opencosmo.io.io import open_single_dataset
from opencosmo.io.mpi import get_all_keys
from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.mpi import get_comm_world, get_mpi

if TYPE_CHECKING:
    import astropy.units as u  # type: ignore
    from astropy.coordinates import SkyCoord
    from astropy.cosmology import Cosmology
    from astropy.table import Table

    from opencosmo.column.column import ColumnMask
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.io.io import OpenTarget
    from opencosmo.io.schema import Schema
    from opencosmo.parameters.hacc import HaccSimulationParameters
    from opencosmo.spatial import Region


def get_redshift_range(datasets: Sequence[Dataset | Lightcone]):
    redshift_ranges = list(map(get_single_redshift_range, datasets))
    min_z = min(rr[0] for rr in redshift_ranges)
    max_z = max(rr[1] for rr in redshift_ranges)

    return (min_z, max_z)


def get_single_redshift_range(dataset: Dataset | Lightcone):
    if isinstance(dataset, Lightcone):
        return dataset.z_range
    redshift_range = dataset.header.lightcone["z_range"]
    if redshift_range is not None:
        return redshift_range
    step_zs = dataset.header.simulation["step_zs"]
    step = dataset.header.file.step
    assert step is not None
    min_redshift = step_zs[step]
    max_redshift = step_zs[step - 1]
    return (min_redshift, max_redshift)


def is_in_range(dataset: Dataset, z_low: float, z_high: float):
    z_range = dataset.header.lightcone["z_range"]
    if z_range is None:
        z_range = get_single_redshift_range(dataset)
    if z_high < z_range[0] or z_low > z_range[1]:
        return False
    return True


def sort_table(table: Table, column: str, invert: bool):
    column_data = table[column]
    if invert:
        column_data = -column_data
    indices = np.argsort(column_data)
    for name in table.columns:
        table[name] = table[name][indices]
    return table


def take_from_sorted(
    lightcone: "Lightcone", sort_by: str, invert: bool, n: int, at: str | int
):
    column = np.concatenate(
        [ds.select(sort_by).get_data("numpy") for ds in lightcone.values()]
    )
    if invert:
        column = -column
    sort_index = np.argsort(column)
    if at == "start":
        sort_index = sort_index[:n]
    elif at == "end":
        sort_index = sort_index[-n:]
    elif isinstance(at, int):
        if at + n > len(sort_index) or at < 0:
            raise ValueError(
                "Requested a range that is outside the size of this dataset!"
            )
        sort_index = sort_index[at : at + n]

    sorted_indices = np.sort(sort_index)
    return sorted_indices


def order_by_redshift_range(datasets: dict[str, Dataset]):
    redshift_ranges = {
        key: get_single_redshift_range(ds) for key, ds in datasets.items()
    }
    sorted_ranges = sorted(redshift_ranges.items(), key=lambda item: item[1][0])
    output = OrderedDict()
    for name, _ in sorted_ranges:
        output[name] = datasets[name]
    return output


def combine_adjacent_datasets_mpi(
    ordered_datasets: dict[str, dict[str, Dataset]],
    min_dataset_size,
):
    MIN_DATASET_SIZE = 100_000
    comm = get_comm_world()
    MPI = get_mpi()
    all_dataset_steps = get_all_keys(ordered_datasets, comm)
    assert comm is not None and MPI is not None
    rs = 0
    output_datasets: dict[str, list[dict[str, Dataset]]] = OrderedDict()
    for step in all_dataset_steps:
        if rs == 0:
            current_key = step
            output_datasets[current_key] = []

        if step not in ordered_datasets:
            rs += comm.allreduce(0, MPI.SUM)
        else:
            length = sum(len(ds) for ds in ordered_datasets[step].values())
            rs += comm.allreduce(length)
            output_datasets[current_key].append(ordered_datasets[step])

        if rs > MIN_DATASET_SIZE:
            rs = 0

    output = OrderedDict()
    for step, datasets in output_datasets.items():
        step_output = defaultdict(list)
        for ds_group in datasets:
            for ds_type, ds in ds_group.items():
                step_output[ds_type].append(ds)
        output[step] = step_output

    return output


def combine_adjacent_datasets(
    ordered_datasets: dict[str, Dataset] | dict[str, dict[str, Dataset]],
    min_dataset_size=100_000,
):
    is_single = isinstance(next(iter(ordered_datasets.values())), Dataset)
    datasets: dict[str, dict[str, Dataset]]
    if is_single:
        assert all(isinstance(ds, Dataset) for ds in ordered_datasets.values())
        datasets = {key: {"data": ds} for key, ds in ordered_datasets.items()}  # type: ignore
    else:
        assert all(isinstance(ds, dict) for ds in ordered_datasets.values())
        datasets = ordered_datasets  # type: ignore

    if get_comm_world() is not None:
        return combine_adjacent_datasets_mpi(datasets, min_dataset_size)

    running_sum = 0

    current_key = next(iter(ordered_datasets.keys()))
    output_datasets: dict[str, list[dict[str, Dataset]]] = OrderedDict(
        {current_key: []}
    )

    for key, step_datasets in datasets.items():
        if running_sum < min_dataset_size:
            running_sum += sum(len(ds) for ds in step_datasets.values())
            output_datasets[current_key].append(step_datasets)
            continue
        current_key = key
        output_datasets[current_key] = [step_datasets]
        running_sum = sum(len(ds) for ds in step_datasets.values())

    # We have list of dicts, go to dict of lists
    output = OrderedDict()
    for step, step_datasets_ in output_datasets.items():
        step_output = defaultdict(list)
        for ds_group in step_datasets_:
            for ds_type, ds in ds_group.items():
                step_output[ds_type].append(ds)
        output[step] = step_output

    return output


def with_redshift_column(dataset: Dataset):
    """
    Ensures a column exists called "redshift" which contains the redshift of the objects
    in the lightcone.
    """
    if "redshift" in dataset.columns:
        return dataset

    elif "fof_halo_center_a" in dataset.columns:
        z_col = 1 / oc.col("fof_halo_center_a") - 1
        return dataset.with_new_columns(redshift=z_col)
    elif "redshift_true" in dataset.columns:
        z_col = 1 * oc.col("redshift_true")
        return dataset.with_new_columns(redshift=z_col)
    raise ValueError(
        "Unable to find a redshift or scale factor column for this lightcone dataset"
    )


class Lightcone(dict):
    """
    A lightcone contains two or more datasets that are part of a lightcone. Typically
    each dataset will cover a specific redshift range. The Lightcone object
    hides these details, providing an API that is identical to the standard
    Dataset API. Additionally, the lightcone contains some convinience functions
    for standard operations.

    Lightcones can be nested. In this case, the top level will split the datasets
    up by step, while the second level will split the datasets up by type. This nested
    scheme (at present) is used for Diffsky catalogs, which may contain both cores
    and synthetic cores that need to be adddressed (and more importantly, written)
    seperately from one another.
    """

    def __init__(
        self,
        datasets: Mapping[Any, Dataset | Lightcone],
        z_range: Optional[tuple[float, float]] = None,
        hidden: Optional[set[str]] = None,
        ordered_by: Optional[tuple[str, bool]] = None,
    ):
        datasets = {
            k: with_redshift_column(ds) if isinstance(ds, Dataset) else ds
            for k, ds in datasets.items()
        }
        self.update(datasets)
        z_range = (
            z_range
            if z_range is not None
            else get_redshift_range(list(datasets.values()))
        )

        columns: set[str] = reduce(
            lambda left, right: left.union(set(right.columns)), self.values(), set()
        )
        if len(columns) != len(next(iter(self.values())).columns):
            raise ValueError("Not all lightcone datasets have the same columns!")
        header = next(iter(self.values())).header
        self.__header = header.with_parameter("lightcone/z_range", z_range)

        if hidden is None:
            hidden = set()

        self.__hidden = hidden
        self.__ordered_by = ordered_by

    def __repr__(self):
        """
        A basic string representation of the dataset
        """
        length = len(self)

        if len(self) < 10:
            repr_ds = self
            table_head = ""
        else:
            repr_ds = self.take(10, at="start")
            table_head = "First 10 rows:\n"

        table_repr = repr_ds.data.__repr__()
        # remove the first line
        table_repr = table_repr[table_repr.find("\n") + 1 :]
        z_range = self.z_range
        head = (
            f"OpenCosmo Lightcone Dataset (length={length}, "
            f"{z_range[0]} < z < {z_range[1]})\n"
        )
        cosmo_repr = f"Cosmology: {self.cosmology.__repr__()}" + "\n"
        return head + cosmo_repr + table_head + table_repr

    def __len__(self):
        return sum(len(ds) for ds in self.values())

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        for dataset in self.values():
            try:
                dataset.close()
            except ValueError:
                continue

    @property
    def header(self) -> OpenCosmoHeader:
        """
        The header associated with this dataset.

        OpenCosmo headers generally contain information about the original data this
        dataset was produced from, as well as any analysis that was done along
        the way.

        Returns
        -------
        header: opencosmo.header.OpenCosmoHeader

        """
        return self.__header

    @property
    def columns(self) -> list[str]:
        """
        The names of the columns in this dataset.

        Returns
        -------
        columns: list[str]
        """
        cols = next(iter(self.values())).columns
        cols = list(filter(lambda col: col not in self.__hidden, cols))
        return cols

    @cached_property
    def descriptions(self) -> dict[str, Optional[str]]:
        """
        Return the descriptions (if any) of the columns in this lightcone as a dictonary.
        Columns without a description will be included in the dictionary with a value
        of None

        Returns
        -------

        descriptions : dict[str, str | None]
            The column descriptions
        """
        descriptions = next(iter(self.values())).descriptions
        descriptions = dict(
            filter(lambda kv: kv[0] not in self.__hidden, descriptions.items())
        )
        return descriptions

    @property
    def cosmology(self) -> Cosmology:
        """
        The cosmology of the simulation this dataset is drawn from as
        an astropy.cosmology.Cosmology object.

        Returns
        -------
        cosmology: astropy.cosmology.Cosmology
        """
        return self.__header.cosmology

    @property
    def dtype(self) -> str:
        """
        The data type of this dataset.

        Returns
        -------
        dtype: str
        """
        return self.__header.file.data_type

    @property
    def region(self) -> Region:
        """
        The region this dataset is contained in. If no spatial
        queries have been performed, this will be the entire
        simulation box for snapshots or the full sky for lightcones

        Returns
        -------
        region: opencosmo.spatial.Region

        """
        return next(iter(self.values())).region

    @property
    def simulation(self) -> HaccSimulationParameters:
        """
        The parameters of the simulation this dataset is drawn
        from.

        Returns
        -------
        parameters: opencosmo.parameters.hacc.HaccSimulationParameters
        """
        return self.__header.simulation

    @property
    def z_range(self):
        """
        The redshift range of this lightcone.

        Returns
        -------
        z_range: tuple[float, float]
        """

        return self.__header.lightcone["z_range"]

    def get_data(self, output="astropy", unpack: bool = False):
        """
        Get the data in this dataset as an astropy table/column or as
        numpy array(s). Note that a dataset does not load data from disk into
        memory until this function is called. As a result, you should not call
        this function until you have performed any transformations you plan to
        on the data.

        You can get the data in two formats, "astropy" (the default) and "numpy".
        "astropy" format will return the data as an astropy table with associated
        units. "numpy" will return the data as a dictionary of numpy arrays. The
        numpy values will be in the associated unit convention, but no actual
        units will be attached.

        If the dataset only contains a single column, it will be returned as an
        astropy.table.Column or a single numpy array.

        Parameters
        ----------
        output: str, default="astropy"
            The format to output the data in. Currently supported are "astropy", "numpy",
            "pandas", "polars", and "arrow"

        Returns
        -------
        data: Table | Column | dict[str, ndarray] | ndarray
            The data in this dataset.
        """
        verify_format(output)

        data = [ds.get_data(unpack=unpack) for ds in self.values()]
        data_with_length = [d for d in data if len(d) > 0]
        if len(data_with_length) == 0:
            return data[0]

        table = vstack(data_with_length, join_type="exact")

        if self.__ordered_by is not None:
            table.sort(self.__ordered_by[0], reverse=self.__ordered_by[1])

        table.remove_columns(self.__hidden)
        if output != "astropy":
            return convert_data(dict(table), output)
        elif len(table.columns) == 1:
            return next(iter(dict(table).values()))

        return table

    @property
    def data(self):
        """
        Return the data in the dataset in astropy format. The value of this
        attribute is equivalent to the return value of
        :code:`Dataset.get_data("astropy")`.

        Returns
        -------
        data : astropy.table.Table or astropy.table.Column
            The data in the dataset.

        """
        return self.get_data("astropy")

    @classmethod
    def open(cls, targets: list[OpenTarget], **kwargs):
        datasets: dict[int, dict[str, Dataset]] = defaultdict(dict)

        for target in targets:
            group_name = target.group.name.split("/")[-1]
            group_name = group_name.lstrip(f"{target.header.file.step}_")
            ds = open_single_dataset(target, bypass_lightcone=True)
            step = target.header.file.step
            assert step is not None
            datasets[step][group_name] = ds

        output: dict[int, Dataset | Lightcone] = {}
        for key, ds_group in datasets.items():
            if len(ds_group) == 1:
                output[key] = next(iter(ds_group.values()))
            else:
                output[key] = Lightcone(ds_group)

        if not all(type(ds) is oc.Dataset for ds in output.values()) and not all(
            type(ds) is Lightcone for ds in output.values()
        ):
            raise ValueError()

        return cls(output)

    def with_redshift_range(self, z_low: float, z_high: float):
        """
        Restrict this lightcone to a specific redshift range. Lightcone datasets will
        always contain a column titled "redshift." This function is always operates on
        this column.

        This function also updates the value in
        :py:meth:`Lightcone.z_range <opencosmo.collection.Lightcone.z_range>`,
        so you should always use it rather than filteringo n the column directly.
        """
        z_range = self.__header.lightcone["z_range"]
        if z_high < z_low:
            z_high, z_low = z_low, z_high

        if z_high < z_range[0] or z_low > z_range[1]:
            raise ValueError(
                f"This lightcone only ranges from z = {z_range[0]} to z = {z_range[1]}"
            )

        elif z_low == z_high:
            raise ValueError("Low and high values of the redshift range are the same!")
        new_datasets = {}
        for key, dataset in self.items():
            if not is_in_range(dataset, z_low, z_high):
                continue
            new_dataset = dataset.filter(
                oc.col("redshift") > z_low, oc.col("redshift") < z_high
            )
            if len(new_dataset) > 0:
                new_datasets[key] = new_dataset
        return Lightcone(
            new_datasets, (z_low, z_high), self.__hidden, self.__ordered_by
        )

    def __map(
        self,
        method,
        *args,
        hidden: Optional[set[str]] = None,
        mapped_arguments: dict[str, dict[str, Any]] = {},
        construct: bool = True,
        **kwargs,
    ):
        """
        This type of collection will only ever be constructed if all the underlying
        datasets have the same data type, so it is always safe to map operations
        across all of them.
        """
        output = {}
        hidden = hidden if hidden is not None else self.__hidden
        for ds_name, dataset in self.items():
            dataset_mapped_arguments = {
                arg_name: args[ds_name] for arg_name, args in mapped_arguments.items()
            }
            output[ds_name] = getattr(dataset, method)(
                *args, **kwargs, **dataset_mapped_arguments
            )

        if construct:
            return Lightcone(output, self.z_range, hidden, self.__ordered_by)
        return output

    def __map_attribute(self, attribute):
        return {k: getattr(v, attribute) for k, v in self.items()}

    def make_schema(self, name: str = "", _min_size=100_000) -> Schema:
        datasets = order_by_redshift_range(self)
        for key in datasets:
            if isinstance(datasets[key], Lightcone):
                datasets[key] = dict(datasets[key])
        output_datasets = combine_adjacent_datasets(
            datasets, min_dataset_size=_min_size
        )
        children = {}

        for step, datasets in output_datasets.items():
            if len(datasets) == 0:
                stack_lightcone_datasets_in_schema(datasets, None, None)
                continue

            all_datasets = list(chain(*tuple(lst for lst in datasets.values())))
            header_zrange = get_redshift_range(all_datasets)
            my_zrange = self.z_range
            zrange = (
                max(header_zrange[0], my_zrange[0]),
                min(header_zrange[1], my_zrange[1]),
            )

            child_schemas = stack_lightcone_datasets_in_schema(datasets, step, zrange)
            child_schemas = {
                f"{step}_{name}": schema for name, schema in child_schemas.items()
            }
            children.update(child_schemas)

        return make_schema(name, FileEntry.LIGHTCONE, children=children)

    def bound(self, region: Region, select_by: Optional[str] = None):
        """
        Restrict the dataset to some subregion. The subregion will always be evaluated
        in the same units as the current dataset. For example, if the dataset is
        in the default "comoving" unit convention, positions are always in units of
        comoving Mpc. However Region objects themselves do not carry units.
        See :doc:`spatial_ref` for details of how to construct regions.

        Parameters
        ----------
        region: opencosmo.spatial.Region
            The region to query.

        Returns
        -------
        dataset: opencosmo.Dataset
            The portion of the dataset inside the selected region

        Raises
        ------
        ValueError
            If the query region does not overlap with the region this dataset resides
            in
        AttributeError:
            If the dataset does not contain a spatial index
        """
        return self.__map("bound", region, select_by)

    def cone_search(self, center: tuple | SkyCoord, radius: float | u.Quantity):
        """
        Perform a search for objects within some angular distance of some
        given point on the sky. This is a convinience function around
        :py:meth:`bound <opencosmo.Lightcone.bound>` and is exactly
        equivalent to

        .. code-block:: python

            region = oc.make_cone(center, radius)
            ds = ds.bound(region)

        Parameters
        ----------
        center: tuple | SkyCoord
            The center of the region to search. If a tuple and no units are provided
            assumed to be RA and Dec in degrees.

        radius: float | astropy.units.Quantity
            The angular radius of the region to query. If no units are provided,
            assumed to be degrees.

        Returns
        -------
        new_lightcone: opencosmo.Lightcone
            The rows in this lightcone that fall within the given region.

        """
        region = oc.make_cone(center, radius)
        return self.bound(region)

    def evaluate(
        self,
        func: Callable,
        format: str = "astropy",
        vectorize=False,
        insert=True,
        **evaluate_kwargs,
    ):
        """
        Iterate over the rows in this collection, apply `func` to each, and collect
        the result as new columns in the dataset. You may also choose to simply return thevalues
        instead of inserting them as a column

        This function is the equivalent of :py:meth:`with_new_columns <opencosmo.Lightcone.with_new_columns>`
        for cases where the new column is not a simple algebraic combination of existing columns. Unlike
        :code:`with_new_columns`, this method will evaluate the results immediately and the resulting
        columns will not change under unit transformations.

        The function should take in arguments with the same name as the columns in this dataset that
        are needed for the computation, and should return a dictionary of output values.
        The dataset will automatically select the needed columns to avoid unnecessarily reading
        data from disk. The new columns will have the same names as the keys of the output dictionary
        See :ref:`Evaluating On Datasets` for more details.

        If vectorize is set to True, the full columns will be pased to the dataset. Otherwise,
        rows will be passed to the function one at a time.

        This function behaves identically to :py:meth:`Dataset.evaluate <opencosmo.Dataset.evaluate>`

        Parameters
        ----------
        func: Callable
            The function to evaluate on the rows in the dataset.

        format: str, default = "astropy"
            The format of the data that is provided to your function. If "astropy", will be a dictionary of
            astropy quantities. If "numpy", will be a dictionary of numpy arrays. Note that
            this method does not support all the formats available in :py:meth:`get_data <opencosmo.Lightcone.get_data>`

        vectorize: bool, default = False
            Whether to provide the values as full columns (True) or one row at a time (False)

        insert: bool, default = True
            If true, the data will be inserted as a column in this dataset. Otherwise the data will be returned.

        Returns
        -------
        dataset : Lightcone
            The new lightcone dataset with the evaluated column(s)
        """
        kwargs, iterable_kwargs = prepare_kwargs(len(self), evaluate_kwargs)
        iterable_kwargs_by_dataset = {}
        indices = np.cumsum(np.fromiter((len(ds) for ds in self.values()), dtype=int))[
            :-1
        ]
        for name, arr in iterable_kwargs.items():
            splits = np.array_split(arr, indices)
            iterable_kwargs_by_dataset[name] = dict(zip(self.keys(), splits))

        result = self.__map(
            "evaluate",
            func=func,
            format=format,
            vectorize=vectorize,
            insert=insert,
            mapped_arguments=iterable_kwargs_by_dataset,
            construct=insert,
            **kwargs,
        )
        if next(iter(result.values())) is None:
            return

        if insert:
            assert isinstance(result, Lightcone)
            return result

        keys = next(iter(result.values())).keys()
        output = {}
        for key in keys:
            output[key] = np.concatenate([r[key] for r in result.values()])
        return output

    def filter(self, *masks: ColumnMask, **kwargs) -> Self:
        """
        Filter the dataset based on some criteria. See :ref:`Querying Based on Column
        Values` for more information.

        Parameters
        ----------
        *masks : Mask
            The masks to apply to dataset, constructed with :func:`opencosmo.col`

        Returns
        -------
        dataset : Dataset
            The new dataset with the masks applied.

        Raises
        ------
        ValueError
            If the given  refers to columns that are
            not in the dataset, or the  would return zero rows.

        """
        return self.__map("filter", *masks, **kwargs)

    def rows(self) -> Generator[dict[str, float | u.Quantity], None, None]:
        """
        Iterate over the rows in the dataset. Rows are returned as a dictionary
        For performance, it is recommended to first select the columns you need to
        work with.

        Yields
        -------
        row : dict
            A dictionary of values for each row in the dataset with units.
        """
        yield from chain.from_iterable(v.rows() for v in self.values())

    def select(self, columns: str | Iterable[str]) -> Self:
        """
        Create a new dataset from a subset of columns in this dataset.

        Parameters
        ----------
        columns : str or list[str]
            The column or columns to select.

        Returns
        -------
        dataset : Dataset
            The new dataset with only the selected columns.

        Raises
        ------
        ValueError
            If any of the given columns are not in the dataset.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = set(columns)
        hidden = self.__hidden

        if "redshift" not in columns:
            columns.add("redshift")
            hidden = hidden.union({"redshift"})

        if self.__ordered_by is not None and self.__ordered_by[0] not in columns:
            columns.add(self.__ordered_by[0])
            hidden = hidden.union({self.__ordered_by[0]})

        return self.__map("select", columns, hidden=hidden)

    def drop(self, columns: str | Iterable[str]) -> Self:
        """
        Produce a new dataset by dropping columns from this dataset.

        Parameters
        ----------
        columns : str or list[str]
            The column or columns to drop.

        Returns
        -------
        dataset : Dataset
            The new dataset without the dropped columns

        Raises
        ------
        ValueError
            If any of the given columns are not in the dataset.
        """
        if isinstance(columns, str):
            columns = [columns]

        dropped_columns = set(columns)
        current_columns = set(self.columns)
        if missing := dropped_columns.difference(current_columns):
            raise ValueError(
                f"Tried to drop columns that are not in this dataset: {missing}"
            )
        kept_columns = current_columns - dropped_columns
        return self.select(kept_columns)

    def take(self, n: int, at: str = "random") -> "Lightcone":
        """
        Create a new dataset from some number of rows from this dataset.

        Can take the first n rows, the last n rows, or n random rows
        depending on the value of 'at'.

        Parameters
        ----------
        n : int
            The number of rows to take.
        at : str
            Where to take the rows from. One of "start", "end", or "random".
            The default is "random".

        Returns
        -------
        dataset : Dataset
            The new dataset with only the selected rows.

        Raises
        ------
        ValueError
            If n is negative or greater than the number of rows in the dataset,
            or if 'at' is invalid.

        """
        if n > len(self):
            raise ValueError(
                "Number of rows to take must be less than number of rows in dataset"
            )
        if at == "random":
            indices = np.random.choice(len(self), n, replace=False)
            indices = np.sort(indices)
            return self.__take_rows(indices)

        elif self.__ordered_by is not None:
            index = take_from_sorted(self, *self.__ordered_by, n=n, at=at)
            return self.__take_rows(index)
        elif at == "start":
            return self.take_range(0, n)
        elif at == "end":
            return self.take_range(len(self) - n, len(self))
        else:
            raise ValueError(
                f'"at" should be one of ("start", "end", "random", got {at}'
            )

    def take_range(self, start: int, end: int):
        """
        Create a new lightcone from a row range in this lightcone. We use standard
        indexing conventions, so the rows included will be start -> end - 1. Because
        lightcones are stacked by redshift, this operation effectively takes a
        redshift range. If you know the exact redshift range you want, use
        :py:meth:`with_redshift_range <opencosmo.Lightcone.with_redshift_range>`.

        Parameters
        ----------
        start : int
            The beginning of the range
        end : int
            The end of the range

        Returns
        -------
        lightcone : opencosmo.Lightcone
            The lightcone with only the specified range of rows.

        Raises
        ------
        ValueError
            If start or end are negative or greater than the length of the dataset
            or if end is greater than start.

        """
        if start < 0 or end > len(self):
            raise ValueError("Got row indices that are out of range!")

        if self.__ordered_by is not None:
            indices = take_from_sorted(self, *self.__ordered_by, end - start, at=start)
            return self.__take_rows(indices)

        ends = np.cumsum(np.fromiter((len(ds) for ds in self.values()), dtype=int))
        starts = np.insert(ends, 0, 0)[:-1]
        clipped_starts = np.clip(starts, a_min=start, a_max=None)
        clipped_ends = np.clip(ends, a_min=None, a_max=end)

        output = {}
        for i, (name, dataset) in enumerate(self.items()):
            if starts[i] == clipped_starts[i] and ends[i] == clipped_ends[i]:
                output[name] = dataset
            elif clipped_starts[i] >= clipped_ends[i]:
                continue
            else:
                output[name] = dataset.take_range(
                    clipped_starts[i] - starts[i], clipped_ends[i] - starts[i]
                )
        return Lightcone(output, self.z_range, self.__hidden, self.__ordered_by)

    def take_rows(self, rows: np.ndarray):
        """
        Take the rows of a lightcone specified by the :code:`rows` argument.
        :code:`rows` should be an array of integers.

        Parameters
        ----------
        rows : np.ndarray[int]
            The indices of the rows to take.

        Returns
        -------
        dataset: The dataset with only the specified rows included

        Raises:
        -------
        ValueError:
            If any of the indices is less than 0 or greater than the length of the
            lightcone.

        """
        rows = np.sort(rows)
        if rows[-1] >= len(self) or rows[0] < 0:
            raise ValueError(
                "Rows must be between 0 and the length of this dataset - 1"
            )
        if self.__ordered_by is not None:
            sort_index = self.__make_sort_index()
            rows = sort_index[rows]
            rows.sort()

        return self.__take_rows(rows)

    def __make_sort_index(self):
        if self.__ordered_by is None:
            return None
        data = np.concatenate(
            [ds.select(self.__ordered_by[0]).get_data("numpy") for ds in self.values()]
        )
        if self.__ordered_by[1]:
            data = -data
        return np.argsort(data)

    def __take_rows(self, rows: np.ndarray):
        """
        Takes rows from this lightcone while ignoring sort. "rows" is assumed to be sorte.
        For internal use only.
        """
        ds_ends = np.cumsum(np.fromiter((len(ds) for ds in self.values()), dtype=int))
        partitions = np.searchsorted(rows, ds_ends)
        splits = np.split(rows, partitions)
        rs = 0
        output = {}
        for split, (name, dataset) in zip(splits, self.items()):
            if len(split) > 0:
                output[name] = dataset.take_rows(split - rs)
            else:
                output[name] = dataset.take(0)  # compatability reasons
            rs += len(dataset)
        return Lightcone(output, self.z_range, self.__hidden, self.__ordered_by)

    def with_new_columns(
        self,
        descriptions: str | dict[str, str] = {},
        **columns: DerivedColumn | np.ndarray | u.Quantity,
    ):
        """
        Create a new dataset with additional columns. These new columns can be derived
        from columns already in the dataset, a numpy array, or an Astropy quantity
        array. When a column is derived from other columns, it will behave
        appropriately under unit transformations. See :ref:`Adding Custom Columns`
        and :py:meth:`Dataset.with_new_columns <opencosmo.Dataset.with_new_columns>`
        for examples.

        Parameters
        ----------
        descriptions : str | dict[str, str], optional
            A description for the new columns. These descriptions will be accessible through
            :py:attr:`Lightcone.descriptions <opencosmo.Lighcone.descriptions>`. If a dictionary,
            should have keys matching the column names.

        ** columns : opencosmo.DerivedColumn | np.ndarray | u.quantity
            The new columns

        Returns
        -------
        dataset : opencosmo.Dataset
            This dataset with the columns added

        """
        derived = {}
        raw = {}
        for name, column in columns.items():
            if isinstance(column, DerivedColumn):
                derived[name] = column
            elif len(column) != len(self):
                raise ValueError(
                    f"New column {name} has length {len(column)} but this dataset "
                    f"has length {len(self)}"
                )
            else:
                raw[name] = column

        if self.__ordered_by is not None:
            sort_index = self.__make_sort_index()
            sort_index = np.argsort(sort_index)
            raw = {name: raw_data[sort_index] for name, raw_data in raw.items()}

        split_points = np.cumsum([len(ds) for ds in self.values()])
        split_points = np.insert(0, 0, split_points)[:-1]
        raw_split = {name: np.split(arr, split_points) for name, arr in raw.items()}
        new_datasets = {}
        for i, (ds_name, ds) in enumerate(self.items()):
            raw_columns = {name: arrs[i] for name, arrs in raw_split.items()}
            columns_input = raw_columns | derived
            new_dataset = ds.with_new_columns(descriptions, **columns_input)
            new_datasets[ds_name] = new_dataset
        return Lightcone(new_datasets, self.z_range, self.__hidden, self.__ordered_by)

    def sort_by(self, column: str, invert: bool = False):
        """
        Sort this dataset by the values in a given column. By default sorting is in
        ascending order (least to greatest). Pass invert = True to sort in descending
        order (greatest to least).

        This can be used to, for example, select largest halos in a given
        dataset:

        .. code-block:: python

            dataset = oc.open("haloproperties.hdf5")
            dataset = dataset
                        .sort_by("fof_halo_mass")
                        .take(100, at="start")

        Parameters
        ----------
        column : str
            The column in the halo_properties or galaxy_properties dataset to
            order the collection by.

        invert : bool, default = False
            If False (the default), ordering will be from least to greatest.
            Otherwise greatest to least.

        Returns
        -------
        result : Dataset
            A new Dataset ordered by the given column.


        """

        if column not in self.columns:
            raise ValueError(f"Column {column} does not exist in this dataset!")
        return Lightcone(dict(self), self.z_range, self.__hidden, (column, invert))

    def with_units(
        self,
        convention: Optional[str] = None,
        conversions: dict[u.Unit, u.Unit] = {},
        **columns: u.Unit,
    ) -> Self:
        r"""
        Create a new lightcone from this one with a different unit convention or
        with certain columns converted to a different compatible unit.

        Unit conversions are always performed after a change of convention, and
        changing conventions clears any existing unit conversions.

        For more, see :doc:`units`.

        .. code-block:: python

            import astropy.units as u

            # this works
            lc = lc.with_units(fof_halo_mass=u.kg)

            # this clears the previous conversion
            lc = lc.with_units("scalefree")

            # This now fails, because the units of masses
            # are Msun / h, which cannot be converted to kg
            lc = lc.with_units(fof_halo_mass=u.kg)

            # this will now work, wince the units of halo mass in the "physical"
            # convention are Msun (no h).
            lc = lc.with_units("physical", fof_halo_mass=u.kg, fof_halo_center_x=u.lyr)

            # Suppose you want your distances in lightyears, but the x coordinate of your
            # halo center in kilometers, for some reason ¯\_(ツ)_/¯
            blanket_conversions = {u.Mpc: u.lyr}
            lc = lc.with_units(conversions = blanket_conversions, fof_halo_center_x = u.km)

        Parameters
        ----------
        convention : str, optional
            The unit convention to use. One of "physical", "comoving",
            "scalefree", or "unitless".

        conversions: dict[astropy.units.Unit, astropy.units.Unit]
            Conversions that apply to all columns in the lightcone with the
            unit given by the key.

        **column_conversions: astropy.units.Unit
            Custom unit conversions for specific columns
            in this dataset.

        Returns
        -------
        lightcone : Lightcone
            The new lightcone with the requested unit convention and/or conversions.
        """
        return self.__map(
            "with_units",
            convention=convention,
            conversions=conversions,
            **columns,
        )
