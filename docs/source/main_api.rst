Main Transformations API
=========================

:code:`opencosmo` provides a simple but powerful API for transforming and querying datasets and collections. Both the main :py:class:`opencosmo.Dataset` type and the various collection types use the same basic vocabulary to describe these transformations, although the details of how they behave will differ. See :doc:`collections` for more info. The main Dataset transformations are:

- :code:`filter`: Filter the dataset based on the value of one more more columns.
- :code:`select`: Select a subset of columns from the dataset.
- :code:`take`: Select a subset of rows from the dataset.
- :code:`with_units`: Change the unit convention of the dataset.
- :code:`crop`: Limit the dataset to a given spatial region (not yet implemented).

Each of these transformations is returns a new dataset or collection with the transformations applied. Because transformations are applied lazily, chaining them together is efficient:

.. code-block:: python

   import opencosmo as oc

   # Load a dataset
   ds = oc.load("haloproperties.hdf5")

   # Apply a series of transformations
   ds = ds.with_units("scalefree")
   ds = ds.filter(oc.col("fof_halo_mass") > 1e13)
   ds = ds.select(["fof_halo_mass", "fof_halo_center_x", "fof_halo_center_y", "fof_halo_center_z"])
   ds = ds.take(100, at="random")
   ds = ds.with_units("physical")
   data = ds.data

In this example, we are we are applying a cut in halo mass using scalefree coordinates, meaning this filter will include all galaxies over 1e13 Msun/h. We then select a subset of the columns and transform them into physical units, removing the factors of h in the final values. See below for more information about unit conventions.

When writing queries like this, it can feel a bit redundant to write :code:`ds = ds.transform(...)` over and over. In practice it is often more readable to simply apply transformations on top of each other:

.. code-block:: python

   ds = oc.load("haloproperties.hdf5")

   ds = ds
      .with_units("scalefree")
      .filter(oc.col("fof_halo_mass") > 1e13)
      .select(["fof_halo_mass", "fof_halo_center_x", "fof_halo_center_y", "fof_halo_center_z"])
      .take(100, at="random")
      .with_units("physical")

   data = ds.data

Note that if you're working in a Jupyter notebook, you'll need to use the line continuation character to split the query across multiple lines:

.. code-block:: python

   ds = oc.load("haloproperties.hdf5")

   ds = ds \
      .with_units("scalefree") \
      .filter(oc.col("fof_halo_mass") > 1e13) \
      .select(["fof_halo_mass", "fof_halo_center_x", "fof_halo_center_y", "fof_halo_center_z"]) \
      .take(100, at="random") \
      .with_units("physical")

   data = ds.data

You are also free to create multiple derivative datasets from the same original dataset:

.. code-block:: python

   ds = oc.load("haloproperties.hdf5")

   ds1 = ds
      .filter(oc.col("fof_halo_mass") > 1e13, os.col("fof_halo_mass") < 1e14)
      .with_units("phsyical")
      .select(["fof_halo_mass", "fof_halo_cdelta"])

   ds2 = ds
      .filter(oc.col("fof_halo_mass") > 1e14)
      .with_units("physical")
      .select(["fof_halo_mass", "fof_halo_cdelta"])

   data1 = ds1.data
   data2 = ds2.data

Because transformations are evaluated lazily, you can have many derivative datasets without incurring a large memory overhead.


Unit Conventions
----------------

The :code:`with_units` transformation is used to change the unit convention of the dataset. :code:`opencosmo` supports the following unit conventions:

- :code:`unitless`: The dataset is read without applying any units
- :code:`scalefree`: The dataset is in "scale-free" units, meaning all lengths are in comoving Mpc/h and all masses are in Msun/h. This is the unit convention that the raw values are stored in.
- :code:`comoving`: Factors of `h` are absorbed into the values, but positions and velocities still use comoving coordinates.
- :code:`physical`: Factors of `h` are absorbed into the values, and positions and velocities are converted to physical coordinates.

When you initially load a dataset, it always uses the "comoving" unit convention. You can change this at any time on any dataset or collection by simply calling :code:`with_units` with the desired unit convention.

Filtering
---------

Filters operate on columns of a given dataset and return a new dataset that only contains the rows that satisfy the filter. Filters are constructed using the :py:func:`opencosmo.col` function, so they can be constructed independently of any single dataset. Available filters include:

- Equality: :code:`col("column_name") == value`
- Inequality: :code:`col("column_name") != value`
- Greater than: :code:`col("column_name") > value`
- Greater than or equal to: :code:`col("column_name") >= value`
- Less than: :code:`col("column_name") < value`
- Less than or equal to: :code:`col("column_name") <= value`
- Membership: :code:`col("column_name").isin([value1, value2, ...])`

When passed to a dataset with the :py:meth:`opencosmo.Dataset.filter` transformation, numerical filters are always applied in the unit convention that is currently active. For a newly constructed dataset, this is always the "comoving" unit convention. See above for an example of applying a filter between unit conventions.




The behavior of filters on collections depends on the collection type. See the :doc:`collections` page for more information.


Selecting Columns
-----------------

For small datasets, it is usually not an issue to request all the columns in a given dataset. However for large datasets, loading everything into memory is slow and consumes singificant quantities of memory. We can use the :py:meth:`opencosmo.Dataset.select` transformation to select only the subset of columns from the dataset that are useful for our analysis. Select transformations can be applied sequentially, in which case the second select will only work if it contains columns that were selected in the first select. For example:

.. code-block:: python

   ds = oc.load("haloproperties.hdf5")

   ds = ds
      .select(["fof_halo_mass", "fof_halo_center_x", "fof_halo_center_y", "fof_halo_center_z"])
      .select(["fof_halo_mass", "fof_halo_center_x"]) 
      # This is fine


.. code-block:: python

   ds = oc.load("haloproperties.hdf5")

   ds = ds
      .select(["fof_halo_mass", "fof_halo_center_x", "fof_halo_center_y", "fof_halo_center_z"])
      .select(["fof_halo_mass", "sod_halo_cdelta"]) 
      # This will raise an error, because sod_halo_cdelta was not in the first select

Filters and selects generally behave as you might expect. If you select *after* filtering, the resulting dataset will only have the columns that were selected for the rows that passed the filter. If you select *before* filtering, the filter can only use columns that were included in the select. For example, this works:

.. code-block:: python

   import opencosmo as oc
   ds = oc.load("haloproperties.hdf5")

   ds = ds
      .select(["fof_halo_mass", "fof_halo_center_x", "fof_halo_center_y", "fof_halo_center_z"])
      .filter(oc.col("fof_halo_mass") > 1e13)


as does this:

.. code-block:: python

   ds = oc.load("haloproperties.hdf5")

   ds = ds
      .filter(oc.col("fof_halo_mass") > 1e13)
      .select(["fof_halo_center_x", "fof_halo_center_y", "fof_halo_center_z"])
      # This is also fine


but this will raise an error:

.. code-block:: python

   ds = oc.load("haloproperties.hdf5")

   ds = ds
      .select(["fof_halo_center_x", "fof_halo_center_y", "fof_halo_center_z"])
      .filter(oc.col("fof_halo_mass") > 1e13)
      # fof_halo_mass is not in the dataset when "filter" is called.


Taking Rows
-----------

The :py:meth:`opencosmo.Dataset.take` transformation is used to select a subset of rows from a dataset. The :code:`at` argument can be used to specify how the rows are selected. The available options are:

- :code:`at="random"`: Select a random subset of n rows from the dataset.
- :code:`at="start"`: Select the first n rows from the dataset.
- :code:`at="end"`: Select the last n rows from the dataset.

As with the `select` transformations, `take` transformations can be chained together. However you cannot take more rows than are present in the dataset:

.. code-block:: python

   ds = oc.load("haloproperties.hdf5")

   ds = ds
      .take(100, at="random")
      .take(500, at="random")
      # This will raise an error



You can also take a range of rows with :py:meth:`opencosmo.Dataset.take_range`. As with all other transformations, this creates a new dataset so the following is valid:  

.. code-block:: python

   ds = oc.load("haloproperties.hdf5")

   ds = ds
      .take_range(500, 1000)
      .take(100, at="start")

This will take the rows 500-1000 from the original dataset, and then take the first 100 rows from that new dataset. The original dataset is unchanged.

Iterating Over Rows:
--------------------
If you want to work row-by-row, you can always iterate over the dataset with :py:meth:`opencosmo.Dataset.rows`

.. code-block:: python

   ds = oc.load("haloproperties.hdf5")

   for row in ds.rows():
      # Do something with the row
      print(row["fof_halo_mass"], row["fof_halo_center_x"])

At each iteration, the row will be a dictionary of values for the specified rows with units applied. If you only need a subset of the columns, consider using :py:meth:`opencosmo.Dataset.select` to select only those columns before iteration.

