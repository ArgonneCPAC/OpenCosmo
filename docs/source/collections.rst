Working with collections
========================

Multiple datasets can be grouped together into *collections.* A collection allows you to perform high-level operations across many datasets at a time, and link related datasets together. In general, collections implement the same :doc:`main_api` as the :py:class:`oc.Dataset` class, with some important caveats (see below).

Types of Collections
--------------------

OpenCosmo currently implements two collection types. Simulation collections hold datasets of the same type from several simulations. Object collections hold multiple data types from a single collection, grouped by object. Simulation collections may also hold object collections. Both collection types implement a dictionary-like interface for accessing the underlying datasets when needed.

In general, you do not need to do anything special to read data into a collection. OpenCosmo will automatically detect if a file has a collection in it.

Simulation Collections
----------------------

SimulationCollections implement an identical API to the :py:class:`oc.Dataset` or :py:class:`oc.link.StructureCollection` it holds. All operations will automatically be mapped over all datasets held by the collection, which must be of the same type. 

Structure Collections
---------------------

A Structure Collection contains datasets of multiple types that are linked together by they structure (halo or galaxy) they are associated with in the simulation. Structure collections always contain at least one *properties* dataset, and one or more particle or profile dataset. For example, a structure collection could contain halo properties and the associated dark matter particles. A structure collection makes it easy to iterate over these objects to perform operations:

.. code-block:: python

   import opencosmo as oc
   data = oc.link.open_linked_files("haloproperties.hdf5", "haloparticles.hdf5")
   for halo, particles in data.objects():
      # do work

At each iteration of the loop, "halo" will be a dictionary of the properties of a singe halo, while "particles" will be a dictionary of :py:class:`oc.Dataset`, one for each particle species.

If you don't need all the particle species, you can always select one or multiple that you actually care about when you do the iteration:
for halo, dm_particles in data.objects(["dm_particles"]):
   # do work

Where dm_particles will now be a dataset containing the dark matter particles for the given halo.

Transformations on Structure Collections
----------------------------------------

Structure Collections implement the :doc:main_api, but with some important differences to behavior.

Transformations Apply to Properties (by default)

Structure Collections always contain a property dataset that contains the high-level information about the structures in the dataset. If the collection contains a dataset with dark matter halo properties...

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

You can always select subests of the columns in any of the individual datasets will keeping them housed in the collection

.. code-block:: python

   import opencosmo as oc
   data = oc.open("my_collection.hdf5")
   data = data.select(["gal_mass", "star_mass"]), dataset="galaxy_properties")

If the "dataset" argument is not provided, the selection will be performed on the property dataset.

Transforming to a different unit convention is identical to :py:meth:`oc.Dataset.with_units` and always applies to all datasets in the collection:

.. code-block:: python

   import opencosmo as oc
   data = oc.open("my_collection.hdf5")
   data = data.with_units("scalefree")

Much like filter operations, take operations operate on the primary Properties dataset.







