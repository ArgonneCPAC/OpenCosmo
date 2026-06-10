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

You can add new columns derived from existing ones using :py:meth:`opencosmo.col` expressions. The preferred way to do this is with :py:meth:`Dataset.select <opencosmo.Dataset.select>`, using ``"*"`` to retain all existing columns alongside the new ones. :py:meth:`Dataset.with_new_columns <opencosmo.Dataset.with_new_columns>` is also available and behaves identically.

Combining Columns into New Columns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use :py:meth:`opencosmo.col` to combine columns to create new columns. Because these new columns are created from pre-existing columns, they will behave as expected under transformations such as a change in unit convention.

.. code-block:: python

   fof_halo_speed_sqrd = oc.col("fof_halo_com_vx") ** 2 + oc.col("fof_halo_com_vy") ** 2 + oc.col("fof_halo_com_vz") ** 2
   fof_halo_ke = 0.5 * oc.col("fof_halo_mass") * fof_halo_speed_sqrd

   # Preferred: select with "*" keeps all existing columns
   ds = ds.select("*", fof_halo_ke=fof_halo_ke)

   # Also works
   ds = ds.with_new_columns(fof_halo_ke=fof_halo_ke)

   ds = ds.with_units("physical")

You can add multiple derived columns in a single call:

.. code-block:: python

   fof_halo_speed_sqrd = oc.col("fof_halo_com_vx") ** 2 + oc.col("fof_halo_com_vy") ** 2 + oc.col("fof_halo_com_vz") ** 2
   fof_halo_ke = 0.5 * oc.col("fof_halo_mass") * fof_halo_speed_sqrd
   fof_halo_p = oc.col("fof_halo_mass") * fof_halo_speed_sqrd ** 0.5

   ds = ds.select("*", fof_halo_ke=fof_halo_ke, fof_halo_p=fof_halo_p)

:py:meth:`opencosmo.Dataset.select` checks to ensure that the columns you are using already exist in the dataset and that the units of the various columns match. For example

.. code-block:: python

   # Forgot to square the x velocity!
   fof_halo_speed_sqrd = oc.col("fof_halo_com_vx") + oc.col("fof_halo_com_vy") ** 2 + oc.col("fof_halo_com_vz") ** 2
   fof_halo_ke = 0.5 * oc.col("fof_halo_mass") * fof_halo_speed_sqrd

   ds = ds.select("*", fof_halo_ke=fof_halo_ke)

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

        dataset = dataset.select("*", fof_halo_com_velocity=total_halo_velocity)

You can find a list of available column combinations in the :ref:`column API reference <Provided Column Combinations>`


Adding Columns Manually
^^^^^^^^^^^^^^^^^^^^^^^

Both :py:meth:`select <opencosmo.Dataset.select>` and :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>` accept numpy arrays and astropy quantity arrays. These arrays must be the same length as the dataset they are being added to. Note that if your data has units, these units will not be transformed under calls to :py:meth:`with_units <opencosmo.Dataset.with_units>`.

.. code-block:: python

        random_data = np.random.randint(0, 1000, size=len(ds)) * u.s
        dataset = dataset.select("*", random_time=random_data)


Creating New Columns in Collections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calls to :py:meth:`opencosmo.StructureCollection.with_new_columns` must explicitly say which dataset the column is being added to:

.. code-block:: python

   import opencosmo as oc
   ds = oc.open("haloproperties.hdf5", "haloparticles.hdf5")
   fof_halo_speed_sqrd = oc.col("fof_halo_com_vx") + oc.col("fof_halo_com_vy") ** 2 + oc.col("fof_halo_com_vz") ** 2
   fof_halo_ke = 0.5 * oc.col("fof_halo_mass") * fof_halo_speed_sqrd
   fof_halo_p = oc.col("fof_halo_mass") * fof_halo_speed_sqrd ** 0.5

   ds = ds.with_new_columns(dataset="halo_properties", fof_halo_ke=fof_halo_ke, fof_halo_p=fof_halo_p)

Calls to :py:meth:`opencosmo.SimulationCollection.with_new_columns` will always apply the new columns to all the datasets in the collection. Because of this, passing a numpy array or astropy quantity object will generally not work, since the length of the datasets within the collection will be different. You can always add a column to a single dataset in the collections by passing the optional :code:`dataset` parameter:

.. code-block:: python

   random_data = np.random.randint(0, 100, len(collection["simulation_a"]))
   collection = collection.with_new_columns(dataset="simulation_a", random_data=random_data)


Column Scalar Reductions
------------------------

Columns support reduction methods that collapse all values in the column into a single scalar quantity. These are useful both for retrieving summary statistics and for building normalized or standardized columns.

The available reduction methods on a :py:meth:`Column <opencosmo.col>` are:

- ``.mean()`` — arithmetic mean
- ``.min()`` — minimum value
- ``.max()`` — maximum value
- ``.std()`` — standard deviation
- ``.var()`` — variance
- ``.median()`` — median
- ``.sum()`` — sum
- ``.quantile(q)`` — quantile at level ``q`` (0 ≤ q ≤ 1)

Scalars in Column Arithmetic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scalar reductions can be used directly in column arithmetic. This is useful for normalization patterns:

.. code-block:: python

   import opencosmo as oc

   ds = oc.open("haloproperties.hdf5")
   m = oc.col("fof_halo_mass")

   # Z-score normalization
   ds = ds.select("*", zscore=(m - m.mean()) / m.std())

   # Min-max scaling to [0, 1]
   ds = ds.select("*", scaled=(m - m.min()) / (m.max() - m.min()))

   # Robust scaling using the interquartile range
   iqr = m.quantile(0.75) - m.quantile(0.25)
   ds = ds.select("*", robust=(m - m.median()) / iqr)

The scalar is evaluated over the rows present in the dataset at the time ``get_data()`` is called, so any prior ``filter()`` or ``bound()`` calls will affect the result.

Retrieving Scalar Values
^^^^^^^^^^^^^^^^^^^^^^^^^

You can retrieve scalar quantities directly by passing only scalar expressions as keyword arguments to :py:meth:`select <opencosmo.Dataset.select>`:

.. code-block:: python

   ds = oc.open("haloproperties.hdf5")

   # Single scalar — get_data() returns an astropy Quantity directly
   min_mass = ds.select(min_mass=oc.col("fof_halo_mass").min()).get_data()

   # Multiple scalars — get_data() returns a dict of Quantities
   stats = ds.select(
       min_mass=oc.col("fof_halo_mass").min(),
       max_mass=oc.col("fof_halo_mass").max(),
       mean_mass=oc.col("fof_halo_mass").mean(),
   ).get_data()

To get bare numpy scalars instead of astropy Quantities, pass ``format="numpy"`` to ``get_data()``:

.. code-block:: python

   min_mass = ds.select(min_mass=oc.col("fof_halo_mass").min()).get_data(format="numpy")

Scalar selections **cannot** be mixed with column selections in a single ``select()`` call. To combine scalar statistics with columnar data, make two separate calls:

.. code-block:: python

   # This raises ValueError:
   ds.select("fof_halo_mass", min_mass=oc.col("fof_halo_mass").min())

   # Do this instead:
   columnar_ds = ds.select("fof_halo_mass", "fof_halo_center_x")
   min_mass = ds.select(min_mass=oc.col("fof_halo_mass").min()).get_data()

Scalar reductions also cannot be passed to ``with_new_columns`` on a :py:class:`StructureCollection <opencosmo.StructureCollection>`. Access the underlying dataset directly and use ``select()`` there:

.. code-block:: python

   collection = oc.open("haloproperties.hdf5", "haloparticles.hdf5")

   # This raises an error:
   # collection.with_new_columns(halo_properties, min_mass=oc.col("fof_halo_mass").min())

   # Do this instead:
   min_mass = collection["halo_properties"].select(
       min_mass=oc.col("fof_halo_mass").min()
   ).get_data()

For scalar reductions across MPI ranks, see :ref:`Working with MPI`.
