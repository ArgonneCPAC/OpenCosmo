from __future__ import annotations

from pathlib import Path

import h5py

try:
    from mpi4py import MPI

    from opencosmo.handler import MPIHandler
except ImportError:
    MPI = None  # type: ignore
import numpy as np

import opencosmo.transformations as t
from opencosmo.dataset.column import ColumnBuilder, get_column_builders
from opencosmo.dataset.mask import Mask, apply_masks
from opencosmo.file import FileExistance, file_reader, file_writer, resolve_path
from opencosmo.handler import InMemoryHandler, OpenCosmoDataHandler, OutOfMemoryHandler
from opencosmo.header import OpenCosmoHeader, read_header, write_header
from opencosmo.spatial import read_tree
from opencosmo.transformations import units as u


def open(file: str | Path, units: str = "comoving") -> Dataset:
    """
    Open a dataset from a file without reading the data into memory.

    The object returned by this function will only read data from the file
    when it is actually needed. This is useful if the file is very large
    and you only need to access a small part of it.

    If you open a file with this dataset, you should generally close it
    when you're done

    .. code-block:: python

        import opencosmo as oc
        ds = oc.open("path/to/file.hdf5")
        # do work
        ds.close()

    Alternatively you can use a context manager, which will close the file
    automatically when you are done with it.

    .. code-block:: python

        import opencosmo as oc
        with oc.open("path/to/file.hdf5") as ds:
            # do work

    Parameters
    ----------
    file : str or pathlib.Path
        The path to the file to open.
    units : str
        The unit convention to use. One of "physical", "comoving",
        "scalefree", or "unitless". The default is "comoving".

    """
    path = resolve_path(file, FileExistance.MUST_EXIST)
    file_handle = h5py.File(path, "r")
    header = read_header(file_handle)
    tree = read_tree(file_handle, header)

    handler: OpenCosmoDataHandler
    if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
        handler = MPIHandler(file_handle, tree=tree, comm=MPI.COMM_WORLD)
    else:
        handler = OutOfMemoryHandler(file_handle, tree=tree)

    base_unit_transformations, transformations = u.get_unit_transformations(
        file_handle["data"], header, units
    )
    column_names = list(str(col) for col in file_handle["data"].keys())
    builders = get_column_builders(transformations, column_names)
    mask = np.ones(len(handler), dtype=bool)

    dataset = Dataset(handler, header, builders, base_unit_transformations, mask)
    return dataset


@file_reader
def read(file: h5py.File, units: str = "comoving") -> Dataset:
    """
    Read a dataset from a file into memory.

    You should use this function if the data are small enough that having
    a copy of it (or a few copies of it) in memory is not a problem. For
    larger datasets, use :py:func:`opencosmo.open`.

    Parameters
    ----------
    file : str or pathlib.Path
        The path to the file to read.
    units : str | None
        The unit convention to use. One of "physical", "comoving",
        "scalefree", or None. The default is "comoving".

    Returns
    -------
    dataset : Dataset
        The dataset read from the file.

    """
    header = read_header(file)
    tree = read_tree(file, header)
    handler = InMemoryHandler(file, tree)
    base_unit_transformations, transformations = u.get_unit_transformations(
        file["data"], header, units
    )
    column_names = list(str(col) for col in file["data"].keys())
    builders = get_column_builders(transformations, column_names)
    mask = np.ones(len(handler), dtype=bool)

    return Dataset(handler, header, builders, base_unit_transformations, mask)


@file_writer
def write(file: h5py.File, dataset: Dataset):
    """
    Write a dataset to a file.

    Parameters
    ----------
    file : str or pathlib.Path
        The path to the file to write to.
    dataset : Dataset
        The dataset to write.

    """
    dataset.write(file)


class Dataset:
    def __init__(
        self,
        handler: OpenCosmoDataHandler,
        header: OpenCosmoHeader,
        builders: dict[str, ColumnBuilder],
        unit_transformations: dict[t.TransformationType, list[t.Transformation]],
        mask: np.ndarray,
    ):
        self.__header = header
        self.__handler = handler
        self.__builders = builders
        self.__base_unit_transformations = unit_transformations
        self.__mask = mask

    def __repr__(self):
        length = np.sum(self.__)
        take_length = length if length < 10 else 10
        repr_ds = self.take(take_length)
        table_repr = repr_ds.data.__repr__()
        # remove the first line
        table_repr = table_repr[table_repr.find("\n") + 1 :]
        head = f"OpenCosmo Dataset (length={length})\n"
        cosmo_repr = f"Cosmology: {self.cosmology.__repr__()}" + "\n"
        table_head = f"First {take_length} rows:\n"
        return head + cosmo_repr + table_head + table_repr

    def __enter__(self):
        # Need to write tests
        return self

    def __exit__(self, *exc_details):
        return self.__handler.__exit__(*exc_details)

    def close(self):
        return self.__handler.__exit__()

    def write(self, file: h5py.File, dataset_name: str = "data") -> None:
        """
        Write the dataset to a file. This should not be called directly for the user.
        The opencosmo.write file writer automatically handles the file context.

        Parameters
        ----------
        file : h5py.File
            The file to write to.
        dataset_name : str
            The name of the dataset in the file. The default is "data".

        """
        if not isinstance(file, h5py.File):
            raise AttributeError(
                "Dataset.write should not be called directly, "
                "use opencosmo.write instead."
            )
        write_header(file, self.__header)
        self.__handler.write(file, self.__mask, self.__builders.keys(), dataset_name)

    @property
    def cosmology(self):
        return self.__header.cosmology

    @property
    def data(self):
        # should rename this, dataset.data can get confusing
        # Also the point is that there's MORE data than just the table
        return self.__handler.get_data(builders=self.__builders, mask=self.__mask)

    def filter(self, *masks: Mask) -> Dataset:
        """
        Filter the dataset based on some criteria.

        Parameters
        ----------
        s : mask
            The s to apply to the dataset.

        Returns
        -------
        dataset : Dataset
            The new dataset with the s applied.

        Raises
        ------
        ValueError
            If the given  refers to columns that are
            not in the dataset, or the  would return zero rows.

        """
        new_mask = apply_masks(self.__handler, self.__builders, masks, self.__mask)
        if np.sum(new_mask) == 0:
            raise ValueError(" would return zero rows.")

        return Dataset(
            self.__handler,
            self.__header,
            self.__builders,
            self.__base_unit_transformations,
            new_mask,
        )

    def select(self, columns: str | list[str]) -> Dataset:
        """
        Select a subset of columns from the dataset.

        Parameters
        ----------
        columns : str or list of str
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

        # numpy compatability
        columns = [str(col) for col in columns]

        try:
            new_builders = {col: self.__builders[col] for col in columns}
        except KeyError:
            known_columns = set(self.__builders.keys())
            unknown_columns = set(columns) - known_columns
            raise ValueError(
                "Tried to select columns that aren't in this dataset! Missing columns "
                + ", ".join(unknown_columns)
            )

        return Dataset(
            self.__handler,
            self.__header,
            new_builders,
            self.__base_unit_transformations,
            self.__mask,
        )

    def with_units(self, convention: str) -> Dataset:
        """
        Transform this dataset to a different unit convention

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
        new_transformations = u.get_unit_transition_transformations(
            convention, self.__base_unit_transformations, self.__header.cosmology
        )
        new_builders = get_column_builders(new_transformations, self.__builders.keys())

        return Dataset(
            self.__handler,
            self.__header,
            new_builders,
            self.__base_unit_transformations,
            self.__mask,
        )

    def collect(self) -> Dataset:
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
        new_handler = self.__handler.collect(self.__builders.keys(), self.__mask)
        return Dataset(
            new_handler,
            self.__header,
            self.__builders,
            self.__base_unit_transformations,
            np.ones(len(new_handler), dtype=bool),
        )

    def take(self, n: int, at: str = "start") -> Dataset:
        """
        Take n rows from the dataset.

        Can take the first n rows, the last n rows, or n random rows
        depending on the value of 'at'.

        Parameters
        ----------
        n : int
            The number of rows to take.
        at : str
            Where to take the rows from. One of "start", "end", or "random".
            The default is "start".

        Returns
        -------
        dataset : Dataset
            The new dataset with only the first n rows.

        Raises
        ------
        ValueError
            If n is negative or greater than the number of rows in the dataset,
            or if 'at' is invalid.

        """
        new_mask = self.__handler.take_mask(n, at, self.__mask)
        if np.sum(new_mask) == 0:
            # This should only happen in an MPI context, so
            # delegate error handling to the user.
            raise ValueError("Filter returned zero rows!")

        return Dataset(
            self.__handler,
            self.__header,
            self.__builders,
            self.__base_unit_transformations,
            new_mask,
        )
