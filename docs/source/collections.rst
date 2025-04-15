Working with collections
========================

Multiple datasets can be grouped together into *collections.* A collection allows you to perform high-level operations across many datasets at a time, and link related datasets together. In general, collections implement the same :doc:`main_api` as the :py:class:`opencosmo.Dataset` class, with some important caveats (see below).

Types of Collections
--------------------

OpenCosmo currently implements two collection types. :py:class:`opencosmo.SimulationCollection` collections hold :py:class:`opencosmo.Dataset` or :py:class:`opencosmo.StructureCollection` of the same type from several simulations. :py:class:`opencosmo.StructureCollection` collections hold multiple data types from a single collection, grouped by object. For example, an :py:class:`opencosmo.StructureCollection` could hold halo properties and the associated dark matter particles. See below for information of how these collections can be used. 

Collections can be opened just like datasets using :py:func:`opencosmo.open`, and written with :py:func:`opencosmo.write`.

Simulation Collections
----------------------

SimulationCollections implement an identical API to the :py:class:`opencosmo.Dataset` or :py:class:`opencosmo.StructureCollection` it holds. All operations will automatically be mapped over all datasets held by the collection, which will always be of the same type. See the documentation for those classes for more information 

Structure Collections
---------------------

A Structure Collection contains datasets of multiple types that are linked together by they structure (halo or galaxy) they are associated with in the simulation. Structure collections always contain at least one *properties* dataset, and one or more particle or profile dataset. For example, a structure collection could contain halo properties and the associated dark matter particles. A structure collection makes it easy to iterate over these objects to perform operations:

.. code-block:: python

   import opencosmo as oc
   data = oc.open("haloparticles.hdf5")
   for halo, particles in data.objects():
      print(halo, particles)

At each iteration of the loop, "halo" will be a dictionary of the properties of a singlee halo (with units), while "particles" will be a dictionary of :py:class:`oc.Dataset`, one for each particle species. If there is only one particles specie in the collection, :code:`particles` will simply be a dataset.

If you don't need all the particle species, you can always select one or multiple that you actually care about when you do the iteration:

.. code-block:: python

   for halo, dm_particles in data.objects(["dm_particles"]):
      # do work

Where :code:`dm_particles` will now be a dataset containing the dark matter particles for the given halo. Because the dataset(s) in :code:`dm_particles` are just regular :py:class:`opencosmo.Dataset` objects, you can use all the standard transformations from the :doc:`main_api`.

Transformations on Structure Collections
----------------------------------------

Structure Collections implement the :doc:`main_api`, but with some important differences to behavior.

**Filters Apply to the Property Dataset**

Structure Collections always contain a property dataset that contains the high-level information about the structures in the dataset. Filters by default will always be applied on this dataset. For most collections this will be a halo properties dataset.

For example, calling "filter" on the structure collection will always operate on columns in the propeties dataset. For example, suppose you have a large collection of halos and their associated particles and you want to work only on halos greater than 10^13 m_sun:

.. code-block:: python

   import opencosmo as oc
   data = oc.open("my_collection.hdf5")
   data = data.filter(oc.col("fof_halo_mass") > 1e13)
   for halo, particles in data.objects():
      # do work

Filtering on non-property datasets is not supported. If your collection contains both a halo properties dataset and a galaxy properties dataset, you can filter based on the galaxy properties by passing an additional argument like so:

.. code-block:: python

   import opencosmo as oc
   data = oc.open("my_collection.hdf5")
   data = data.filter(oc.col("gal_mass") > 1e11, dataset="galaxy_properties")

However this comes with an important caveat. Filtering based on properties of a galaxy removes any halo that does not contain any a galaxy that meets the threshold. If a halo hosts multiple galaxies and at least one meets the criteria, all galaxies in the halo will be retained. 

**Select Can Be Made on a Per-Dataset Basis**

You can always select subests of the columns in any of the individual datasets while keeping them housed in the collection

.. code-block:: python

   import opencosmo as oc
   data = oc.open("my_collection.hdf5")
   data = data.select(["x", "y", "z"]), dataset="dm_particles")

If the "dataset" argument is not provided, the selection will be performed on the property dataset.

**Unit Transformations Apply to All Datasets**

Transforming to a different unit convention is identical to :py:meth:`opencosmo.Dataset.with_units` and always applies to all datasets in the collection:

.. code-block:: python

   import opencosmo as oc
   data = oc.open("my_collection.hdf5")
   data = data.with_units("scalefree")


**Take Operations Take Structure**

Calling :py:meth:`opencosmo.StructureCollection.take` will create a new :py:class:`StructureDataset` with the number of structures specifiedin the take operation. This means the following operation will behave as you might expect:

.. code-block:: python
   
   import opencosmo as oc
   ds = oc.open("my_collection.hdf5")
   ds = ds.take(10)

   for halo, particles in ds.objects():
      # this loop iterate over 10 halos








