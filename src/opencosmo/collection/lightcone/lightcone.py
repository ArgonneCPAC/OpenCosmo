from functools import reduce
from itertools import chain
from typing import Any, Callable, Generator, Iterable, Optional, Self

import astropy.units as u  # type: ignore
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore
from astropy.cosmology import Cosmology  # type: ignore
from astropy.table import Column, Table, vstack  # type: ignore

import opencosmo as oc
from opencosmo.dataset import Dataset
from opencosmo.dataset.column import ColumnMask, DerivedColumn
from opencosmo.evaluate import prepare_kwargs
from opencosmo.header import OpenCosmoHeader
from opencosmo.index import SimpleIndex
from opencosmo.io.io import OpenTarget, open_single_dataset
from opencosmo.io.schemas import LightconeSchema
from opencosmo.parameters.hacc import HaccSimulationParameters
from opencosmo.spatial import Region


def get_redshift_range(datasets: list[Dataset]):
    redshift_ranges = [ds.header.lightcone["z_range"] for ds in datasets]
    if all(rr is not None for rr in redshift_ranges):
        min_redshift = min(rr[0] for rr in redshift_ranges)
        max_redshift = max(rr[1] for rr in redshift_ranges)

    else:
        steps = np.fromiter((ds.header.file.step for ds in datasets), dtype=int)
        step_zs = datasets[0].header.simulation["step_zs"]
        min_step = np.min(steps)
        max_step = np.max(steps)

        min_redshift = step_zs[max_step]
        max_redshift = step_zs[min_step - 1]
    return (min_redshift, max_redshift)


def is_in_range(dataset: Dataset, z_low: float, z_high: float):
    z_range = dataset.header.lightcone["z_range"]
    if z_range is None:
        z_range = get_redshift_range([dataset])
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


def make_indices_for_sort(
    lightcone: "Lightcone", sort_by: str, invert: bool, n: int, at: str
):
    column = np.concatenate(
        [ds.select(sort_by).get_data("numpy") for ds in lightcone.values()]
    )
    if invert:
        column = -column
    sorted_indices = np.argsort(column)
    if at == "start":
        sorted_indices = sorted_indices[:n]
    else:
        sorted_indices = sorted_indices[-n:]

    sorted_indices = np.sort(sorted_indices)
    index = SimpleIndex(sorted_indices)
    ranges = np.fromiter((len(ds) for ds in lightcone.values()), dtype=int)
    ranges = np.insert(ranges, 0, 0)
    slices = np.cumsum(index.n_in_range(ranges[:-1], ranges[1:]))
    slices = np.insert(slices, 0, 0)

    output_indices = []
    for i in range(len(slices) - 1):
        output_index = sorted_indices[slices[i] : slices[i + 1]] - ranges[i]
        output_indices.append(SimpleIndex(output_index))
    return output_indices


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


class Lightcone(dict):
    """
    A lightcone contains two or more datasets that are part of a lightcone. Typically
    each dataset will cover a specific redshift range. The Lightcone object
    hides these details, providing an API that is identical to the standard
    Dataset API. Additionally, the lightcone contains some convinience functions
    for standard operations.
    """

    def __init__(
        self,
        datasets: dict[str, Dataset],
        z_range: Optional[tuple[float, float]] = None,
        hidden: Optional[set[str]] = None,
        ordered_by: Optional[tuple[str, bool]] = None,
    ):
        datasets = {k: with_redshift_column(ds) for k, ds in datasets.items()}
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

    def get_data(self, output="astropy"):
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

        This method does not cache data. Calling "get_data" always reads data
        from disk, even if you have already called "get_data" in the past.
        You can use :py:attr:`Dataset.data <opencosmo.Lightcone.data>` to return
        data and keep it in memory.

        Parameters
        ----------
        output: str, default="astropy"
            The format to output the data in

        Returns
        -------
        data: Table | Column | dict[str, ndarray] | ndarray
            The data in this dataset.
        """
        if output not in {"astropy", "numpy"}:
            raise ValueError(f"Unknown output type {output}")

        data = [ds.get_data(unpack=False) for ds in self.values()]
        table = vstack(data, join_type="exact")
        if self.__ordered_by is not None:
            table.sort(self.__ordered_by[0], reverse=self.__ordered_by[1])

        table.remove_columns(self.__hidden)

        if len(table.colnames) == 1:
            table = next(table.itercols())

        if output == "numpy":
            if isinstance(table, (u.Quantity, Column)):
                return table.value
            else:
                return {name: col.value for name, col in table.items()}

        return table

    @property
    def data(self):
        """
        Return the data in the dataset in astropy format. The value of this
        attribute is equivalent to the return value of
        :code:`Dataset.get_data("astropy")`. However data retrieved via this
        attribute will be cached, meaning further calls to
        :py:attr:`Dataset.data <opencosmo.Lightcone.data>` should be instantaneous.

        However there is one caveat. If you modify the table, those modifications will
        persist if you later request the data again with this attribute. Calls to
        :py:meth:`Lightcone.get_data <opencosmo.Lightcone.get_data>` will be unaffected,
        and datasets generated from this dataset will not contain the modifications.
        If you plan to modify the data in this table, you should use
        :py:meth:`Lightcone.with_new_columns <opencosmo.Lightcone.with_new_columns>`.


        Returns
        -------
        data : astropy.table.Table or astropy.table.Column
            The data in the dataset.

        """
        return self.get_data("astropy")

    @classmethod
    def open(cls, targets: list[OpenTarget], **kwargs):
        datasets: dict[str, Dataset] = {}

        for target in targets:
            ds = open_single_dataset(target)
            if not isinstance(ds, Lightcone) or len(ds.keys()) != 1:
                raise ValueError(
                    "Lightcones can only contain datasets (not collections)"
                )
            if target.group.name != "/":
                key = target.group.name.split("/")[-1]
            else:
                key = f"{target.header.file.step}_{target.header.file.data_type}"
            datasets[key] = next(iter(ds.values()))

        return cls(datasets)

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

    def make_schema(self) -> LightconeSchema:
        schema = LightconeSchema()
        for name, dataset in self.items():
            ds_schema = dataset.make_schema()
            ds_schema.header = ds_schema.header.with_parameter(
                "lightcone/z_range", self.z_range
            )
            schema.add_child(ds_schema, name)
        return schema

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
        The dataset will automatically selected the needed columns to avoid reading unnecessarily reading
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
            astropy quantities. If "numpy", will be a dictionary of numpy arrays.

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
            columns.add("redshift")
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
            rs = 0
            output = {}
            indices = np.random.choice(len(self), n, replace=False)
            indices = np.sort(indices)
            for key, ds in self.items():
                indices_into_ds = (
                    indices[(indices >= rs) & (indices < rs + len(ds))] - rs
                )
                output[key] = ds.take(len(indices_into_ds))
                rs += len(ds)
            return Lightcone(
                output, self.z_range, hidden=self.__hidden, ordered_by=self.__ordered_by
            )
        output = {}
        rs = 0
        if self.__ordered_by is not None:
            indices = make_indices_for_sort(self, *self.__ordered_by, n=n, at=at)
            output = {
                k: v.with_index(indices[i]) for i, (k, v) in enumerate(self.items())
            }
            return Lightcone(output, self.z_range, self.__hidden, self.__ordered_by)

        if at == "start":
            iter = self.items()
        elif at == "end":
            iter = reversed(self.items())  # type: ignore

        for name, ds in iter:
            if len(ds) < n - rs:
                output[name] = ds
                rs += len(ds)
            else:
                output[name] = ds.take(n - rs, at=at)
                break
        if at == "end":
            output = {k: v for k, v in reversed(output.items())}
        return Lightcone(output, self.z_range, self.__hidden, self.__ordered_by)

    def with_new_columns(self, **columns: DerivedColumn | np.ndarray | u.Quantity):
        """
        Create a new dataset with additional columns. These new columns can be derived
        from columns already in the dataset, a numpy array, or an Astropy quantity
        array. When a column is derived from other columns, it will behave
        appropriately under unit transformations. See :ref:`Adding Custom Columns`
        and :py:meth:`Dataset.with_new_columns <opencosmo.Dataset.with_new_columns>`
        for examples.

        Parameters
        ----------
        ** columns : opencosmo.DerivedColumn

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

        split_points = np.cumsum([len(ds) for ds in self.values()])
        split_points = np.insert(0, 0, split_points)[:-1]
        raw_split = {name: np.split(arr, split_points) for name, arr in raw.items()}
        new_datasets = {}
        for i, (ds_name, ds) in enumerate(self.items()):
            raw_columns = {name: arrs[i] for name, arrs in raw_split.items()}
            columns_input = raw_columns | derived
            new_dataset = ds.with_new_columns(**columns_input)
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

    def with_units(self, convention: str) -> Self:
        """
        Create a new dataset from this one with a different unit convention.

        Parameters
        ----------
        convention : str
            The unit convention to use. One of "physical", "comoving",
            "scalefree", or "unitless".

        Returns
        -------
        dataset : Dataset
            The new dataset with the requested unit convention.

        """
        return self.__map("with_units", convention)

    def collect(self) -> "Lightcone":
        """
        Given a dataset that was originally opend with opencosmo.open,
        return a dataset that is in-memory as though it was read with
        opencosmo.read.

        This is useful if you have a very large dataset on disk, and you
        want to filter it down and then close the file.

        For example:

        .. code-block:: python

            import opencosmo as oc
            with oc.open("path/to/file.hdf5") as file:
                ds = file.(ds["sod_halo_mass"] > 0)
                ds = ds.select(["sod_halo_mass", "sod_halo_radius"])
                ds = ds.collect()

        The selected data will now be in memory, and the file will be closed.

        If working in an MPI context, all ranks will recieve the same data.
        """
        datasets = {k: v.collect() for k, v in self.items()}
        return Lightcone(datasets, self.z_range, self.__hidden, self.__ordered_by)
