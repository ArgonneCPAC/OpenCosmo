Working with Columns
====================

You can use the toolkit to query your data based on the value of a column, or create new columns based on combinations of columns that already exist in the dataset.

Querying Based on Column Values
-------------------------------

Querying your dataset based on the value of a single column is straightforward:

.. code-block:: python

   # select halos with mass greather than 1e13 Msun
   import opencosmo as oc
   ds = oc.open("haloproperties.hdf5")
   query = oc.col("fof_halo_mass") > 1e14
   ds = data.filter(query)

Because the query is contructed outside the dataset itself, you are free to use it across several datasets at the same time. Because queries create new datasets, you can query the same dataset multiple times easily:

.. code-block:: python

   ds = oc.open("haloproperties.hdf5")
   query_high = oc.col("fof_halo_mass") < 5e13
   query_low = oc.col("fof_halo_mass") > 1e13
   ds_low = data.filter(query_low)
   ds_high = data.filter(query_high)

You can also combine multiple queries to create more specific datasets:

.. code-block:: python

   # select halos with mass between 1e13 and 5e13 Msun
   lower_bound = oc.col("fof_halo_mass") > 1e13
   upper_bound = oc.col("fof_halo_mass") < 5e13
   ds_bounded = ds.filter(lower_bound, upper_bound)

As well as multiple queries across multiple columns.

.. code-block:: python

   # select high-concentration halos between mass of 1e13 and 5e13 
   lower_bound = oc.col("fof_halo_mass") > 1e13
   upper_bound = oc.col("fof_halo_mass") < 5e13
   c_bound = oc.col("sod_halo_cdelta") > 3
   ds_bounded = ds.filter(lower_bound, upper_bound)

The value in the query is always evaluated in the unit conventions of the dataset. For example, the following two queries will find different rows, even if the `ds_scalefree` dataset is transformed back to comoving units after the fact:

.. code-block:: python

   import opencosmo as oc
   ds = oc.open("haloproperties.hdf5")
   query = oc.col("fof_halo_mass") > 1e14

   ds_scalefree = data.with_units("scalefree").filter(query)
   ds_comoving = data.filter(query)


For more information, see :ref:`Unit Conventions`.


Creating New Columns
--------------------

You can also use :py:meth:`oc.col` to combine columns to create new columns. Because these new columns are created from pre-existing colums, they will behave as expected under transformations such as a change in unit convention.

.. code-block:: python

   fof_halo_speed_sqrd = oc.col("fof_halo_com_vx") ** 2 + oc.col("fof_halo_com_vy") ** 2 + oc.col("fof_halo_com_vz") ** 2
   fof_halo_ke = 0.5 * oc.col("fof_halo_mass") * fof_halo_speed_sqrd
   
   ds = ds.with_new_columns(fof_halo_ke = fof_halo_ke)
   ds = ds.with_units("physical")

:py:meth:`opencosomo.dataset.with_new_columns` checks to ensure that the columns you are using already exist in the dataset. But it does not check that the mathematical operation you are attempting to perform is valid until the data is actually requested. For example:

.. code-block:: python

   # Forgot to square the x velocity!
   fof_halo_speed_sqrd = oc.col("fof_halo_com_vx") + oc.col("fof_halo_com_vy") ** 2 + oc.col("fof_halo_com_vz") ** 2
   fof_halo_ke = 0.5 * oc.col("fof_halo_mass") * fof_halo_speed_sqrd
   
   ds = ds.with_new_columns(fof_halo_ke = fof_halo_ke)
   ds = ds.with_units("physical")

The code above will run without errors. But as soon as the actual data is requested:

.. code-block:: python

   data = ds.data


you will get an error. 

.. code-block:: text

        ValueError: To add and subtract columns, units must be the same!

This behavior will be updated in a future version of the library to throw the error at the `with_new_columns` call.
