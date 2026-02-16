Performance Tips
================

:code:`opencosmo` is designed to handle datasets on any scale, from smaller catalogs you work with on your local machine to much larger catalogs housed on supercomputers. However there are some best-practices to keep in mind if you want the best possible performances, especially with very large datasets.

Don't Request Data Until You Actually Need It
---------------------------------------------

For the most part, :code:`opencosmo` queries are `lazily evaluated <https://en.wikipedia.org/wiki/Lazy_evaluation>`_. This means that :code:`opencosmo` only reads data when it absolutely needs to, such as when a user requests it with :meth:`Dataset.get_data <opencosmo.Dataset.get_data>` or writes a dataset to a new file with :meth:`write <opencosmo.write>`.

The main exception to this rule is transformations that filter the dataset based on the value of the columns. If you :meth:`filter <opencosmo.Dataset.filter>` a dataset based on the value of a column, that column will be loaded into memory and your filter will be evaluated immediately. However this only loads the columns necessary to evaluate the filter, so it is usually quite fast. 

As a result, it is usually a good idea to whittle down the dataset as much as possible before requesting data. More specific suggestions can be found below.

Select the Columns You Actually Need
------------------------------------

Loading data from disk into memory will almost always be the most time-consuming part of any query. :code:`opencosmo` allows you to :meth:`select <opencosmo.Dataset.select>` the columns you need at any time. As you may be able to guess by now, a column that is not selected will not be loaded into memory at all. If you have a dataset with many columns and you only needa few, selecting those columns and excluding the others can greatly speed up the time it takes to perform the query.

Preview Data with :meth:`take <opencosmo.Dataset.take>`
-------------------------------------------------------

If you want to test something out, consider taking a small number of rows using the :meth:`take <opencosmo.Dataset.take>` method. Doing this avoids loading data into memory that you do not plan to use for your test. It is easy to create a small test dataset without losing access to the full dataset:

.. code-block:: python

   import opencosmo as oc

   ds = oc.open("haloproperties.hdf5").select(["fof_halo_mass", "sod_halo_mass"])
   test_ds = ds.take(10, at="start")
   print(test_ds.get_data()) # or do other tests
        
   # do additional work with original dataset


Create Multiple Child Datasets from a Single Ancestor
-----------------------------------------------------

:code:`opencosmo` allows you to create many children from a single ancestor dataset. If there are a set of common operations that must be applied to all children, it is generally better to perform them on the ancestor datasets before creating the children. This is especially true for filters and spatial queries. For example, suppose you were interested in looking at the concentration-mass relation for halos in a particular region of your dataset. You want to explore the relationship seperately in two different mass bins. One approach could be to do the following:

.. code-block:: python

        import opencosmo as oc

        ds = oc.open("haloproperties.hdf5")
        region = oc.box((300,300,300), (600,600,6))

        ds_low_mass = ds
            .filter(oc.col("sod_halo_mass") > 1e12, oc.col("sod_halo_mass") < 1e13)
            .bound(region)
            .select(["sod_halo_mass", "sod_halo_cdelta"])

        ds_high_mass = ds
            .filter(oc.col("sod_halo_mass") > 1e13, oc.col("sod_halo_mass") < 1e14)
            .bound(region)
            .select(["sod_halo_mass", "sod_halo_cdelta"])

however this is not optimal, because the spatial query now has to be performed twice. Instead, you can do the following:

.. code-block:: python

        import opencosmo as oc


        region = oc.box((300,300,300), (600,600,6))
        ds = oc.open("haloproperties.hdf5")
        ds = ds
            .bound(region)
            .select(["sod_halo_mass", "sod_halo_cdelta"])

        ds_low_mass = ds.filter(oc.col("sod_halo_mass") > 1e12, oc.col("sod_halo_mass") < 1e13)
        ds_high_mass = ds.filter(oc.col("sod_halo_mass") > 1e13, oc.col("sod_halo_mass") < 1e14)

Of course, you may also want to name the parent dataset something different so that you can keep access to the full un-filtered catalog.





