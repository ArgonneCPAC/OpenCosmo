Main Transformations API
=========================

:code:`opencosmo` provides a simple but powerful API for transforming and querying datasets and collections. Both the main :py:class:`opencosmo.Dataset` type and the various collection types will have these transformations available., although the details of how they behave will differ slightly. Individual collection types may also have additional convinience methods based on their purpose, see :doc:`collections` for more info. The main transformations are:

- :code:`with_units`: Change the unit convention of the dataset or collection.
- :code:`filter`: Filter a dataset based on the value of one more more columns.
- :code:`select`: Select a subset of columns from a dataset.
- :code:`take`: Select a subset of rows from a dataset.
- :code:`sort_by`: Sort a dataset by one of its columns
- :code:`bound`: Limit a dataset or collection to a given spatial region.
- :code:`with_new_columns`: Combine columns in a dataset into a new column with automatic unit handling.
- :code:`evaluate`: Evaluate a computation over all the rows in a dataset or collection.

Each of these transformations is returns a new dataset or collection with the transformations applied. Because transformations are applied lazily, chaining them together is efficient:

.. code-block:: python

   import opencosmo as oc

   # Load a dataset
   ds = oc.open()

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

   ds = oc.open("haloproperties.hdf5")

   ds = ds
      .with_units("scalefree")
      .filter(oc.col("fof_halo_mass") > 1e13)
      .select(["fof_halo_mass", "fof_halo_center_x", "fof_halo_center_y", "fof_halo_center_z"])
      .take(100, at="random")
      .with_units("physical")

   data = ds.data

Note that if you're working in a Jupyter notebook, you'll need to use the line continuation character to split the query across multiple lines:

.. code-block:: python

   ds = oc.open("haloproperties.hdf5")

   ds = ds \
      .with_units("scalefree") \
      .filter(oc.col("fof_halo_mass") > 1e13) \
      .select(["fof_halo_mass", "fof_halo_center_x", "fof_halo_center_y", "fof_halo_center_z"]) \
      .take(100, at="random") \
      .with_units("physical")

   data = ds.data

You are also free to create multiple derivative datasets from the same original dataset:

.. code-block:: python

   ds = oc.open("haloproperties.hdf5")

   low_mass_ds = ds
      .filter(oc.col("fof_halo_mass") > 1e13, os.col("fof_halo_mass") < 1e14)
      .with_units("phsyical")
      .select(["fof_halo_mass", "fof_halo_cdelta"])

   high_mass_ds = ds
      .filter(oc.col("fof_halo_mass") > 1e14)
      .with_units("physical")
      .select(["fof_halo_mass", "fof_halo_cdelta"])

   data1 = ds1.data
   data2 = ds2.data

However you may also be interested in including all data that passes *either* filter in a single dataset. You can combine filters with boolean logic using the & and | operators:

.. code-block:: python

   ds = oc.open("haloproperties.hdf5")

   high_mass_cut = oc.col("fof_halo_mass") > 1e14
   low_mass_cut = oc.col("fof_halo_mass") < 1e12
   low_concentration_cut = oc.col("sod_halo_cdelta") < 5

   my_filter = (high_mass_cut | low_mass_cut) & low_concentration_cut
   filtered_ds = ds.filter(my_filter)


Because transformations are evaluated lazily, you can have many derivative datasets without incurring a large memory overhead.


Unit Conventions
----------------

The :code:`with_units` transformation is used to change the unit convention of the dataset. :code:`opencosmo` supports the following unit conventions:

- :code:`unitless`: The dataset is read without applying any units
- :code:`scalefree`: The dataset is in "scale-free" units, meaning all lengths are in comoving Mpc/h and all masses are in Msun/h. This is the unit convention that the raw values are stored in.
- :code:`comoving`: Factors of `h` are absorbed into the values, but positions and velocities still use comoving coordinates.
- :code:`physical`: Factors of `h` are absorbed into the values, and positions and velocities are converted to physical coordinates.

When you initially load a dataset, it always uses the "comoving" unit convention. You can change this at any time on any dataset or collection by simply calling :code:`with_units` with the desired unit convention. For more information, see :ref:`Working with Units`

Adding Columns
--------------

You can add new columns to a given that are derived from pre-existing columns using the :meth:`oc.col` to construct new columns and passing them to :code:`with_new_columns`. The new columns will inherit the cosmological dependence of the columns they are created from, and can be used throughout the transformations API as usual. 

.. code-block:: python

   ds = oc.open("haloproperties.hdf5")

   fof_halo_vtotal = (oc.col("fof_halo_com_vx")**2 + oc.col("fof_halo_com_vy")**2 + ("fof_halo_com_vz")**2)**(0.5)
   fof_halo_com_p = oc.col("fof_halo_mass") * fof_halo_vtotal

   ds = ds.with_new_columns(fof_halo_com_p = fof_halo_com_p)


The dataset will now contain a "fof_halo_com_p" column that can be used for filtering and selections as usual. Because the column definition was created outside the dataset itself, it can be used across multiple datasets as needed.

You can also simply pass values as a numpy array or astropy quantity:

.. code-block:: python

   import astropy.units as u
   import numpy as np

   random_angle = np.random.uniform(10, 50, len(ds))*u.arcmin
   ds = ds.with_new_columns(angle = random_angle)

Columns can be added to collections as well, but there are some subtelties. See :doc:`collections` for more information.

Filtering
---------

Filters operate on columns of a given dataset and return a new dataset that only contains the rows that satisfy the filter. Filters are constructed using the :meth:`opencosmo.col` function, so they can be constructed independently of any single dataset. Available filters include:

- Equality: :code:`col("column_name") == value`
- Inequality: :code:`col("column_name") != value`
- Greater than: :code:`col("column_name") > value`
- Greater than or equal to: :code:`col("column_name") >= value`
- Less than: :code:`col("column_name") < value`
- Less than or equal to: :code:`col("column_name") <= value`
- Membership: :code:`col("column_name").isin([value1, value2, ...])`

Filters do not need to include units, however a filter with *incorrect* units will raise an error:

.. code-block:: python

   import astropy.units as u
   from astropy.cosmology import units as u
   import opencosmo as oc

   ds = oc.open("haloproperties.hdf5")

   # This will work fine
   min_mass = oc.col("fof_halo_mass") > 1e13 
   ds = ds.filter(min_mass)

   # This will work fine
   min_mass_unitful = oc.col("fof_halo_mass") > 1e13 * u.Msun
   ds = ds.filter(min_mass_unitful)

   # This will fail, because the masses are not in Msun / h
   min_mass_unitful = oc.col("fof_halo_mass") > 1e13 * u.Msun / cu.littleh
   ds = ds.filter(min_mass_unitful)

The behavior of filters on collections depends on the collection type. See the :doc:`collections` page for more information.

Selecting Columns
-----------------

For small datasets, it is usually not an issue to request all the columns in a given dataset. However for large datasets, loading everything into memory is slow and consumes singificant quantities of memory. We can use the :meth:`opencosmo.Dataset.select` transformation to select only the subset of columns that are useful for our analysis. Select transformations can be applied sequentially, in which case the second select will only work if it contains columns that were selected in the first select. For example:

.. code-block:: python

   ds = oc.open("haloproperties.hdf5")

   ds = ds
      .select(["fof_halo_mass", "fof_halo_center_x", "fof_halo_center_y", "fof_halo_center_z"])
      .select(["fof_halo_mass", "fof_halo_center_x"]) 
      # This is fine


.. code-block:: python

   ds = oc.open("haloproperties.hdf5")

   ds = ds
      .select(["fof_halo_mass", "fof_halo_center_x", "fof_halo_center_y", "fof_halo_center_z"])
      .select(["fof_halo_mass", "sod_halo_cdelta"]) 
      # This will raise an error, because sod_halo_cdelta was not in the first select

Filters and selects generally behave as you might expect. If you select *after* filtering, the resulting dataset will only have the columns that were selected for the rows that passed the filter. If you select *before* filtering, the filter can only use columns that were included in the select. For example, this works:

.. code-block:: python

   import opencosmo as oc
   ds = oc.open("haloproperties.hdf5")

   ds = ds
      .select(["fof_halo_mass", "fof_halo_center_x", "fof_halo_center_y", "fof_halo_center_z"])
      .filter(oc.col("fof_halo_mass") > 1e13)


as does this:

.. code-block:: python

   ds = oc.open("haloproperties.hdf5")

   ds = ds
      .filter(oc.col("fof_halo_mass") > 1e13)
      .select(["fof_halo_center_x", "fof_halo_center_y", "fof_halo_center_z"])
      # This is also fine


but this will raise an error:

.. code-block:: python

   ds = oc.open("haloproperties.hdf5")

   ds = ds
      .select(["fof_halo_center_x", "fof_halo_center_y", "fof_halo_center_z"])
      .filter(oc.col("fof_halo_mass") > 1e13)
      # fof_halo_mass is not in the dataset when "filter" is called.


Taking Rows
-----------

The :meth:`opencosmo.Dataset.take` transformation is used to select a subset of rows from a dataset. The :code:`at` argument can be used to specify how the rows are selected. The available options are:

- :code:`at="random"`: Select a random subset of n rows from the dataset (default).
- :code:`at="start"`: Select the first n rows from the dataset.
- :code:`at="end"`: Select the last n rows from the dataset.

As with the `select` transformations, `take` transformations can be chained together. However you cannot take more rows than are present in the dataset:

.. code-block:: python

   ds = oc.open("haloproperties.hdf5")

   ds = ds
      .take(100, at="random")
      .take(500, at="random")
      # This will raise an error

You can also take a range of rows with :meth:`opencosmo.Dataset.take_range`. As with all other transformations, this creates a new dataset so the following is valid:  

.. code-block:: python

   ds = oc.open("haloproperties.hdf5")

   ds = ds
      .take_range(500, 1000)
      .take(100, at="start")

This will take the rows 500-1000 from the original dataset, and then take the first 100 rows from that new dataset. The original dataset is unchanged.

Sorting
-------

You can re-order a dataset based on the value of some column with :meth:`opencosmo.Dataset.sort_by`. By default, this sorts in ascending order (from lowest to highest), however you can sort in descending order by passing :code:`invert = True`.

For example, to get the 100 most massive halos in a given simulation, ordered from most to least massive:

.. code-block:: python

   ds = oc.open("haloproperties.hdf5")
   ds = ds.sort_by("fof_halo_mass", invert=True).take(100, at="start")

Or, to get the 100 *least* massive halos, ordered from least to most massive:


.. code-block:: python

   ds = ds.sort_by("fof_halo_mass").take(100, at="start")

You can also use :py:meth:`take <opencosmo.Dataset.take>` in clever ways to get other results. For example, to get the 100 *most* massive halos but ordered from *least to most massive*:

.. code-block:: python

   ds = ds.sort_by("fof_halo_mass").take(100, at="end")


Spatial Querying
-----------------
OpenCosmo data contains a spatial index which makes it efficient to perform spatial queries on the data. These queries can be performed by defining a region, and then passing it into :meth:`opencosmo.Dataset.bound`:

.. code-block:: python

   ds = oc.open("haloproperties.hdf5")
   region = oc.make_box((20,20,20), (40,40,40))
   bound_ds = ds.bound(region)

For lightcone data, spatial queries are performed using two dimensional regions on the sky. For example:

.. code-block:: python

   import astropy.units as u
   from astropy.coordinates import SkyCoord

   ds = oc.open("lc_haloproperties.hdf5")
   center = SkyCoord(45*u.deg, -30*u.deg)
   radius = 30*u.arcmin
   region = opencosmo.make_cone(center, radius)
   bound_ds = ds.bound(region)

See :doc:`spatial_ref` for more information about constructing regions.

As with other transformations, spatial queries can be chained together to build complex query pipelines. If a given region contains no data, the spatial query will return a dataset with length zero. 

There are some complications that arise when working with spatial queries in an MPI context. See :doc:`mpi` for more details.

Iterating Over Rows
--------------------
If you want to work row-by-row, you can always iterate over the dataset with :meth:`opencosmo.Dataset.rows`

.. code-block:: python

   ds = oc.open("haloproperties.hdf5")

   for row in ds.rows():
      # Do something with the row
      print(row["fof_halo_mass"], row["fof_halo_center_x"])

At each iteration, the row will be a dictionary of values for the specified rows with units applied. If you only need a subset of the columns, consider using :meth:`opencosmo.Dataset.select` to select only those columns before iteration.


Evaluating Complex Expressions
------------------------------

Generally, basic data manipulation is not sufficient for science. We need to fit models and perform complex operations. The :py:meth:`evaluate <opencosmo.Dataset.evaluate>` method can handle the low-level data management, leaving you to focus on building your model. See :doc:`evaluating` for more information.
