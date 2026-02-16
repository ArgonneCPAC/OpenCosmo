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
   ds = ds.filter(query)

Because the query is contructed outside the dataset itself, you are free to use it across several datasets at the same time. Because queries create new datasets, you can query the same dataset multiple times easily:

.. code-block:: python

   ds = oc.open("haloproperties.hdf5")
   query_high = oc.col("fof_halo_mass") < 5e13
   query_low = oc.col("fof_halo_mass") > 1e13
   ds_low = ds.filter(query_low)
   ds_high = ds.filter(query_high)

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
   ds_bounded = ds.filter(lower_bound, upper_bound, c_bound)

The value in the query is always evaluated in the unit conventions of the dataset. For example, the following two queries will find different rows, even if the `ds_scalefree` dataset is transformed back to comoving units after the fact:

.. code-block:: python

   import opencosmo as oc
   ds = oc.open("haloproperties.hdf5")
   query = oc.col("fof_halo_mass") > 1e14

   ds_scalefree = ds.with_units("scalefree").filter(query)
   ds_comoving = ds.filter(query)


For more information, see :ref:`Unit Conventions`.

Querying In Collections
-----------------------

Queries can generally be performed as usual on collections. In a :py:class:`opencosmo.StructureCollection`, the query will be performed on the properties of the structures within the structure collection. For example:

.. code-block:: python

   import opencosmo as oc
   ds = oc.open("haloproperties.hdf5", "haloparticles.hdf5")
   ds = ds.filter(oc.col("fof_halo_mass") > 1e14)

The resultant :py:class:`opencosmo.StructureCollection` will contain only halos with a mass greater than 10^14, along with all their associated particles.

In a :py:class:`opencosmo.SimulationCollection`, the filter will be applied to all datasets inside the collection. 

For more details, see :doc:`collections`.

Adding Custom Columns
---------------------

You can use the :py:meth:`Dataset.with_new_columns <opencosmo.Dataset.with_new_columns>` to add new columns to your data. You can either combine existing columns to create new ones, or create wholly-new columns by providing data as a numpy array or astropy quantity array. 


Combining Columns into New Columns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use :py:meth:`opencosmo.col` to combine columns to create new columns. Because these new columns are created from pre-existing colums, they will behave as expected under transformations such as a change in unit convention.

.. code-block:: python

   fof_halo_speed_sqrd = oc.col("fof_halo_com_vx") ** 2 + oc.col("fof_halo_com_vy") ** 2 + oc.col("fof_halo_com_vz") ** 2
   fof_halo_ke = 0.5 * oc.col("fof_halo_mass") * fof_halo_speed_sqrd
   
   ds = ds.with_new_columns(fof_halo_ke = fof_halo_ke)
   ds = ds.with_units("physical")

You can also always add multiple derived columns in a single call:

.. code-block:: python

   fof_halo_speed_sqrd = oc.col("fof_halo_com_vx") ** 2 + oc.col("fof_halo_com_vy") ** 2 + oc.col("fof_halo_com_vz") ** 2
   fof_halo_ke = 0.5 * oc.col("fof_halo_mass") * fof_halo_speed_sqrd
   fof_halo_p = oc.col("fof_halo_mass") * fof_halo_speed_sqrd ** 0.5
   
   ds = ds.with_new_columns(fof_halo_ke = fof_halo_ke, fof_halo_p = fof_halo_p)

:py:meth:`opencosmo.Dataset.with_new_columns` checks to ensure that the columns you are using already exist in the dataset and that the units of the various columns match. For example

.. code-block:: python

   # Forgot to square the x velocity!
   fof_halo_speed_sqrd = oc.col("fof_halo_com_vx") + oc.col("fof_halo_com_vy") ** 2 + oc.col("fof_halo_com_vz") ** 2
   fof_halo_ke = 0.5 * oc.col("fof_halo_mass") * fof_halo_speed_sqrd
   
   ds = ds.with_new_columns(fof_halo_ke = fof_halo_ke)

you will get an error. 

.. code-block:: text

        opencosmo.units.UnitError: To add and subtract columns, units must be the same!

Built-In Column Combinations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenCosmo provides a number of built-in column combinations that may be useful for a wide variety of analysis. For example, to calculate the total velocity of a halo from its velocity components:

.. code-block:: python

        import opencosmo as oc
        from opencosmo.columns import norm_cols
        
        dataset = oc.open("haloproperties.hdf5")
        total_halo_velocity = norm_cols("fof_halo_com_vx", "fof_halo_com_vy", "fof_halo_com_vz")
        
        dataset = dataset.with_new_columns(fof_halo_com_velocity = total_halo_velocity)
        
You can find a list of available column combinations in the :ref:`column API reference <Provided Column Combinations>`


Adding Columns Manually
^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>` method accepts numpy arrays and astropy quantity arrays. These arrays must be the same length as the dataset they are being added to. Note that if your data has units, these units will not be transformed under calls to :py:meth:`with_units <opencosmo.Dataset.with_units>`. 

.. code-block:: python

        random_data = np.random.randint(0, 1000, size = len(ds)) * u.s
        dataset  = dataset.with_new_columns(random_time = random_data)



Creating New Columns in Collections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calls to :py:meth:`opencosmo.StructureCollection.with_new_columns` must explicitly say which dataset the column is being added to:


.. code-block:: python

   import opencosmo as oc
   ds = oc.open("haloproperties.hdf5", "haloparticles.hdf5")
   fof_halo_speed_sqrd = oc.col("fof_halo_com_vx") + oc.col("fof_halo_com_vy") ** 2 + oc.col("fof_halo_com_vz") ** 2
   fof_halo_ke = 0.5 * oc.col("fof_halo_mass") * fof_halo_speed_sqrd
   fof_halo_p = oc.col("fof_halo_mass") * fof_halo_speed_sqrd ** 0.5

   ds = ds.with_new_columns(dataset="halo_properties", fof_halo_ke = fof_halo_ke, fof_halo_p = fof_halo_pe)

Calls to :py:meth:`opencosmo.SimulationCollection.with_new_columns` will always apply the new columns to all the datasets in the collection. Because of this, passing a numpy array or astropy quantity object will generally not work, since the length of the datasets within the collection will be different. You can always add a column to a single dataset in the collections by passing the optional :code:`dataset` parameter:

.. code-block:: python
   
   random_data = np.random.randint(0, 100, len(collection["simulation_a"]))
   collection = collection.with_new_columns(dataset="simulation_a", random_data=random_data)
