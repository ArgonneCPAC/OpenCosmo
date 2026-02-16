Working with Collections
========================

Multiple datasets can be grouped together into *collections.* A collection allows you to perform high-level operations across many datasets at a time, and link related datasets together. In general, collections implement the same :doc:`main_api` as the :py:class:`opencosmo.Dataset` class, with some important caveats (see below).

Datasets behave a lot like dictionaries. You can get the names of the dataset with ``collection.keys()``, the datasets themselves with ``collection.values()``, or both with ``collection.items()``. A given dataset within the collection can always be accessed with ``collection[dataset_name]``.


Types of Collections
--------------------

OpenCosmo currently implements three collection types. :py:class:`opencosmo.Ligthcone` collections stack datasets in angular coordinates from  different redshift slices into a single dataset-like object. :py:class:`opencosmo.SimulationCollection` collections hold :py:class:`opencosmo.Dataset` or :py:class:`opencosmo.StructureCollection` of the same type from several simulations. :py:class:`opencosmo.StructureCollection` collections hold multiple data types from a single collection, grouped by object. A :py:class:`opencosmo.HealpixMap` holds one or more on-sky maps. See below for information of how these collections can be used. 

Collections can be opened just like datasets using :py:func:`opencosmo.open`, and written with :py:func:`opencosmo.write`.

Lightcones
----------
Lightcones hold datasets from several different redshift steps, which are stacked to form a single dataset. In general, the :py:class:`Lightcone <opencosmo.Lightcone>` API is identical to the standard :py:class:`Dataset <opencosmo.Dataset>` API with the addition of some extra convinience functions. For example:

.. code-block:: python

   import opencosmo as oc
   ds = oc.open("step_600/haloproperties.hdf5", "step_601/haloproperties.hdf5")
   ds = ds.with_redshift_range(0.040, 0.0405)
   print(ds)

.. code-block:: text

        OpenCosmo Lightcone Dataset (length=31487, 0.04 < z < 0.0405)
        Cosmology: FlatLambdaCDM(name=None, H0=<Quantity 67.66 km / (Mpc s)>, Om0=0.3096446816186967, Tcmb0=<Quantity 0. K>, Neff=3.04, m_nu=None, Ob0=0.04897468161869667)
        First 10 rows:
        block    chi     fof_halo_1D_vel_disp fof_halo_center_a fof_halo_center_x ... sod_halo_radius   theta   unique_tag [replication, halo_tag]   redshift
                                km / s                                 Mpc        ...       Mpc
        int64  float32         float32             float32           float32      ...     float32      float32            (int32, int64)             float32
        ----- ---------- -------------------- ----------------- ----------------- ... --------------- --------- ---------------------------------- -----------
            0  120.25088            29.816034         0.9610807         126.16309 ...       -149.2758 1.5622419             (1049601, 60016004986) 0.040495396
            2 120.031685            21.276937         0.9611496         126.76229 ...       -149.2758 1.5499203             (1049601, 60174689392)  0.04042077
            9  119.12508            35.604523         0.9614344         125.27003 ...       -149.2758 1.5308051             (1049601, 58587164999) 0.040112495
            9 119.718056             45.24862        0.96124804        126.242195 ...       -149.2758 1.5266299             (1049601, 59222154598) 0.040314198
            9  119.43137            24.338223         0.9613382        126.190384 ...       -149.2758 1.5229564             (1049601, 59380902001) 0.040216565
           10   119.0838             37.28911         0.9614474         127.76812 ...       -149.2758 1.5297077             (1049601, 61285757392)  0.04009843
           10  120.16473            42.233627         0.9611078         129.43185 ...       -149.2758 1.5301584             (1049601, 62238266991)  0.04046607
           10 119.903854             46.74386        0.96118975         128.91646 ...       -149.2758 1.5263848             (1049601, 62555736594)  0.04037726
           11 119.049706            49.151897        0.96145815         126.78099 ...      0.12496548 1.5141958             (1049601, 60492121202) 0.040086865
           11  119.60659             43.32933         0.9612831        127.099144 ...       -149.2758  1.519284             (1049601, 60492184203)  0.04027629

Structure Collections
---------------------

A Structure Collection contains datasets of multiple types that are linked together by they structure (halo or galaxy) they are associated with in the simulation. Structure collections always contain at least one *properties* dataset, and one or more particle or profile dataset. 

You can always access the individual datasets in the collection just as you would values in a dictionary: 

.. code-block:: python

   import opencosmo as oc
   ds = oc.open("haloproperties.hdf5", "haloparticles.hdf5")
   dm_particles = data["dm_particles"]


However the real power of working with a :py:class:`StructureCollection` is the automatic grouping of these datasets by structure. You can iterate through the structures in the dataset easily:

.. code-block:: python

   import opencosmo as oc
   data = oc.open("haloproperties.hdf5", "haloparticles.hdf5")
   for halo in data.halos():
      print(halo)

At each iteration of the loop, `structure` will contain a dictionary of the properties and datasets associated with the given halo. 

If you don't need all the particle species, you can always select one or multiple that you actually care about when you do the iteration:

.. code-block:: python

   for structure in data.objects(["dm_particles", "gas_particles"]):
      # do work

Where :code:`structure` will now be a dictionary containing three things:

* ``structure["halo_properties"]`` will be a dictionary of the halo properties for the given halo.
* ``structure["dm_particles"]`` will be an :class:`opencosmo.Dataset` with the dark matter particles associated with the halo
* ``structure["gas_particles"]`` will be an :class:`opencosmo.Dataset` with the gas particles associated with the halo

It is also possible for structure collections to contain other structure collections. For example, in a hydro simulation a single halo may contain more than one galaxy. 

.. code-block:: python

   import opencosmo as oc
   ds = oc.open("haloproperties.hdf5", "haloparticles.hdf5", "galaxyproperties.hdf5", "galaxyparticles.hdf5")
   for structure in ds.halos():
        gals_ds = structure["galaxies"]
        for galaxy in gals_ds.galaxies():
                # do work with galaxies.
                
      
You can now iterate through galaxies in the galaxies in the halo just as you would iterate through halos in your full dataset.

Because the structure collection returns regular :class:`opencosmo.Dataset` objects, you can query or transform them further as needed.

Particles take a lot of space on disk, and it is common to only store particles for a certain subset of halos in a simulation. By default, :code:`opencosmo` will filter out structures that do not have any associated particles. If you want to disable this behavior, you can set the :code:`ignore_empty` flag to :code:`False` when you open the collection:

.. code-block:: python

    import opencosmo as oc
    data = oc.open("haloproperties.hdf5", "haloparticles.hdf5", ignore_empty=False)
    for halo in data.halos():
        # Will now include halos that have no particles
        print(halo)





Transformations on Structure Collections
----------------------------------------

Structure Collections implement the :doc:`main_api`, but with some important differences to behavior.

**Filters Apply to the Halo/Galaxy Properties**

Structure Collections always contain a property dataset that contains the high-level information about the structures in the dataset. Filters by default will always be applied on this dataset. 

For example, calling "filter" on the structure collection will always operate on columns in the propeties dataset. For example, suppose you have a large collection of halos and their associated particles and you want to work only on halos greater than 10^13 m_sun:

.. code-block:: python

   import opencosmo as oc
   ds = oc.open("my_collection.hdf5")
   ds = ds.filter(oc.col("fof_halo_mass") > 1e13)
   for halo in data.objects():
      # do work

If your collection contains both a halo properties dataset and a galaxy properties dataset, you can filter based on the galaxy properties by passing an additional argument like so:

.. code-block:: python

   import opencosmo as oc
   ds = oc.open("my_collection.hdf5")
   ds = ds.filter(oc.col("gal_mass") > 1e11, dataset="galaxy_properties")

However this comes with an important caveat. Filtering based on properties of a galaxy removes any halo that does not contain any a galaxy that meets the threshold. If a halo hosts multiple galaxies and at least one meets the criteria, all galaxies in the halo will be retained. 

**Select Can Be Made on a Per-Dataset Basis**

You can always select subests of the columns in any of the individual datasets while keeping them housed in the collection

.. code-block:: python

   import opencosmo as oc
   ds = oc.open("my_collection.hdf5")
   ds = data.select(["x", "y", "z"]), dataset="dm_particles")

If the "dataset" argument is not provided, the selection will be performed on the property dataset.

**Unit Transformations Apply to All Datasets**

Transforming to a different unit convention is identical to :py:meth:`opencosmo.Dataset.with_units` and always applies to all datasets in the collection:

.. code-block:: python

   import opencosmo as oc
   ds = oc.open("my_collection.hdf5")
   ds = ds.with_units("scalefree")


**Take Operations Take Structure**

Calling :py:meth:`opencosmo.StructureCollection.take` will create a new :py:class:`StructureDataset` with the number of structures specified in the take operation. This means the following operation will behave as you might expect:

.. code-block:: python
   
   import opencosmo as oc
   ds = oc.open("my_collection.hdf5")
   ds = ds.take(10)

   for halo, particles in ds.objects():
      # this loop iterate over 10 halos


Healpix Maps
------------
Maps contain pixelized spatial data, integrated over a redshift range, in a single dataset. This may contain one or more different types of data (e.g. x-ray emission or projected density fields) for a given simulation and range. The :py:class:`HealpixMap <opencosmo.HealpixMap>` API is identical to the standard :py:class:`Dataset <opencosmo.Dataset>` API, with a few notable differences. We do not support unit conversions, as we do not assume fine-grained redshift sampling, and instead keep everything in the observed frame. We also provide the data in either Healpix Map format (a numpy array with nested pixelization), or Healsparse format (a sparse implementation which supports partial sky coverage). Some knowledge of Healsparse formats may be useful to work with this data. A simple pixel, value return is demonstrated below. 
 

.. code-block:: python

   import opencosmo as oc
   ds = oc.open(healpix_map_path)
   ds.get_data()

.. code-block:: text

    {'ksz': HealSparseMap: nside_coverage = 64, nside_sparse = 2048, float32, 50331648 valid pixels,
     'tsz': HealSparseMap: nside_coverage = 64, nside_sparse = 2048, float32, 50331648 valid pixels}

.. code-block:: python

   tsz_map = ds.data['tsz']
   pix_list = tsz_map.valid_pixels
   vals = tsz_map.get_values_pix(pix_list)





Simulation Collections
----------------------

SimulationCollections implement an identical API to the :py:class:`opencosmo.Dataset` or :py:class:`opencosmo.StructureCollection` it holds. All operations will automatically be mapped over all datasets held by the collection, which will always be of the same type. See the documentation for those classes for more information 


