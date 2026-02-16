First Steps
===========

To follow this tutorial, download the "haloproperites.hdf5" and "haloparticles.hdf5" files from the `OpenCosmo Google Drive <https://drive.google.com/drive/folders/1CYmZ4sE-RdhRdLhGuYR3rFfgyA3M1mU-?usp=sharing>`_ and set the environment variable. This file contains properties of dark-matter halos from a small hydrodynamical simulation run with HACC. You can easily open the data with the :code:`open` function:

.. code-block:: python

   import opencosmo as oc
   dataset = oc.open("haloproperties.hdf5")
   print(dataset)


.. code-block:: text

   OpenCosmo Dataset (length=237441)
   Cosmology: FlatLambdaCDM(name=None, H0=<Quantity 67.66 km / (Mpc s)>, Om0=0.3096446816186967, Tcmb0=<Quantity 0. K>, Neff=3.04, m_nu=None, Ob0=0.04897468161869667)
   First 10 rows:
   block fof_halo_1D_vel_disp fof_halo_center_x ... sod_halo_sfr unique_tag
                km / s               Mpc        ... solMass / yr
   int32       float32             float32      ...   float32      int64
   ----- -------------------- ----------------- ... ------------ ----------
       0            32.088795         1.4680439 ...       -101.0      21674
       0             41.14525        0.19616994 ...       -101.0      44144
       0             73.82962         1.5071135 ...    3.1447952      48226
       0             31.17231         0.7526525 ...       -101.0      58472
       0            23.038841         5.3246417 ...       -101.0      60550
       0            37.071426         0.5153746 ...       -101.0     537760
       0            26.203058         2.1734374 ...       -101.0     542858
       0              78.7636         2.1477687 ...          0.0     548994
       0             37.12636         6.9660196 ...       -101.0     571540
       0             58.09235          6.072006 ...    1.5439711     576648

The `open` function returns a `Dataset` object, which can retrieve the raw data from disk. It also holds additional information about the data, such as the simulation it was drawn from and the associated cosmology. You can easily access the data and cosmology directly as Astropy objects:

.. code-block:: python

   dataset.get_data()
   dataset.cosmology

The first will return an astropy table of the data, with all associated units already applied. The second will return the astropy cosmology object that represents the cosmology the simulation was run with. 

Basic Querying
--------------

Although you can access data directly, :code:`opencosmo` provides tools for querying and transforming the data in a fully cosmology-aware context. For example, suppose we wanted to plot the concentration-mass relationship for the halos in our simulation above a certain mass. One way to perform this would be as follows:

.. code-block:: python

   dataset = dataset
       .filter(oc.col("fof_halo_mass") > 1e13)
       .take(1000)
       .select(("fof_halo_mass", "sod_halo_cdelta"))

   print(dataset)


.. code-block:: text

   OpenCosmo Dataset (length=1000)
   Cosmology: FlatLambdaCDM(name=None, H0=<Quantity 67.66 km / (Mpc s)>, Om0=0.3096446816186967, Tcmb0=<Quantity 0. K>, Neff=3.04, m_nu=None, Ob0=0.04897468161869667)
   First 10 rows:
    fof_halo_mass   sod_halo_cdelta
       solMass
       float32          float32
   ---------------- ---------------
   11220446000000.0       4.5797048
   17266723000000.0       7.4097505
   51242150000000.0       1.8738283
   70097712000000.0       4.2764015
   51028305000000.0        2.678151
   11960567000000.0       3.9594727
   15276915000000.0        5.793542
   16002001000000.0       2.4318497
   47030307000000.0       3.7146702
   15839942000000.0        3.245569

We could then plot the data, or perform further transformations. 

Data Collections
----------------

This is cool on its own, but the real power of :code:`opencosmo` comes from its ability to work with different data types. Go ahead and download the "haloparticles" file from the `OpenCosmo Google Drive <https://drive.google.com/drive/folders/1CYmZ4sE-RdhRdLhGuYR3rFfgyA3M1mU-?usp=sharing>`_ and try the following:

.. code-block:: python

   import opencosmo as oc
   data = oc.open("haloproperties.hdf5", "haloparticles.hdf5")

This will return a data *collection* that will allow you to query and transform the data as before, but will associate the halos with their particles. 

.. code-block:: python

   structures = data
       .filter(oc.col("fof_halo_mass") > 1e13)
       .take(1000, at="random")

   for halo in structures.halos(["dm_particles", "star_particles"]):
       halo_mass = halo["halo_properties"]["fof_halo_mass"]
       dm_particles = halo["dm_particles"]
       star_particles = halo["star_particles"]
       # do_work

In each iteration, the "halo" object will be a dictionary containing a "halo_properties" dictionary as well as two :py:class:`opencosmo.Dataset` objects, one containing the dark matter particles associated with the halos and the other containing the star particles. Because these are standard datasets, you can perform further transformaions on them as is useful for your analysis.


Core Concepts
-------------

The core design principle of :code:`opencosmo` is `lazy immutability`. Data is not retrieved from disk until it is actaully needed, and dataset and collections cannot be modified once created. This approach leads to a number of performance benefits, and makes it possible to work with data that is much larger than the available memory. However, many data management libraries do not operate in this way so it is good to be aware of how :code:`opencosmo` behaves.

Immutability
~~~~~~~~~~~~

Immutability means that datasets do not change once they are created. Try to spot the problem with the following example:

.. code-block:: python

    import opencosmo as oc
    ds = oc.open("haloproperties.hdf5")
    ds.filter(oc.col("fof_halo_mass") > 1e14)
    data = ds.get_data("numpy")
    assert np.all(data["fof_halo_mass"] > 1e14)

Running this code will give you an assertion errror. When we call :code:`ds.filter`, the :code:`ds` dataset object is not modified. Instead, :code:`ds.filter` returns a new dataset, which is the same as the old dataset but with the new filter applied. Because we don't assign this new dataset to anything, it simply disappears.

We can fix this problem by simply assigning the new dataset to our `ds` variable:

.. code-block:: python

    import opencosmo as oc
    ds = oc.open("haloproperties.hdf5")
    ds = ds.filter(oc.col("fof_halo_mass") > 1e14)
    data = ds.get_data("numpy")
    assert np.all(data["fof_halo_mass"] > 1e14)

This pattern is quite common. Often, we don't care about the original dataset once we've done our filter. In these cases, it is perfectly acceptable to simply discard the original dataset by binding the new dataset to the same variable name.

Because dataset operations return new datasets, we can chain together these operations when we do not care about the intermmediate results:

.. code-block:: python

    import opencosmo as oc
    ds = oc.open("haloproperties.hdf5")

    ds = ds
        .filter(oc.col("fof_halo_mass") > 1e14)
        .select(("fof_halo_mass", "sod_halo_mass", "sod_halo_cdelta"))
        .take(10_000, at="random")
    data = ds.get_data("numpy")

    assert np.all(data["fof_halo_mass"] > 1e14)
    assert len(data) == 3 # Dictionary with 3 columns
    assert len(data["fof_halo_mass"] == 10_000)


However there are some cases where we don't to overwrite the original dataset. We could, if we wanted to, also hold onto the original dataset or create multiple derivative datasets:

.. code-block:: python

    import opencosmo as oc
    ds = oc.open("haloproperties.hdf5")
    ds_high_mass = ds.filter(oc.col("fof_halo_mass") > 1e14)
    ds_low_mass = ds.filter(oc.col("fof_halo_mass") < 1e14)
    data_high_mass = ds_high_mass.get_data("numpy")
    data_low_mass = ds_low_mass.get_data("numpy")
    assert np.all(data_high_mass["fof_halo_mass"] > 1e14)
    assert np.all(data_low_mass["fof_halo_mass"] > 1e14)

Laziness and Caching
~~~~~~~~~~~~~~~~~~~~

`Laziness <https://en.wikipedia.org/wiki/Lazy_evaluation>`_ means putting off the evlauation of some expression until its result is actually needed. In the context of `opencosmo`, this means data remains on disk until it is actually needed. Opening a dataset with :py:meth:`opencosmo.open` doesn't read any data into memory. Instead, it simple ensures that the file conforms to the opencosmo standard and reads enough of the metadata to perform operations and communicate to you about what is in the dataset.

In practice, this means that calling :meth:`get_data <opencosmo.Dataset.get_data>` or writing data to disk with :meth:`opencosmo.write` will almost always be the slowest part of any script. These methods require the toolkit to actually read data from disk, and in some cases even create new columns based on operations like :meth:`with_new_columns <opencosmo.Dataset.with_new_columns>` or :meth:`evaluate <opencosmo.Dataset.evaluate>`. Depending on the quantity of data requested, or the complexity of the relationships between existing columns and new columns, this can take some time. o

However the advanatages of this approach are significant. :code:`opencosmo` has no problem working with datasets that are much too large to fit in memory. You can open such a dataset and perform operations just as you would with a much smaller dataset. So long as the resultant dataset is small enough, everything will work when you call :code:`get_data`. Of course, if you try to read in more data than can fit in memory, :code:`opencosmo` will try to get that data for you even if it crashes your process. 

However this approach leads to some complications. Consider what might happen in the following example.

.. code-block:: python

    import opencosmo as oc
    ds = oc.open("haloproperties.hdf5")
    ds = ds.filter(oc.col("fof_halo_mass") > 1e14)

    data = ds.get_data()

    ds_selected = ds.select(("fof_halo_mass", "sod_halo_mass"))
    data_selected = ds_selected.get_data()

What happens in this situation? Do we load the :code:`fof_halo_mass` and :code:`sod_halo_mass` columns twice? This is inneficient and can quickly lead to memory problems. It's good to avoid reading data until we actually need it, because we will often find that much of the data doesn't end up having to be read at all. But once data `is` read from disk, we want to try to avoid reading it a second time if we can.

:code:`opencosmo` has a caching layer which attempts to avoid reading data twice. In the example above, data read in the first :code:`get_data` call is cached and made available to any downstream datasets. :code:`ds_selected` has access to this cache. Since the :code:`fof_halo_mass` and :code:`sod_halo_mass` columns are already in memory, the final :code:`ds_selected.get_data()` call does not require reading anything from disk or allocating new memory. It simply returns the data that has already been read.

This also works with more subtle cases:

.. code-block:: python

    import opencosmo as oc
    ds = oc.open("haloproperties.hdf5")
    data = ds.get_data()

    ds = ds.filter(oc.col("fof_halo_mass") > 1e14)

    data_filtered = ds.get_data()

Because the all the data required to create :code:`data_filtered` is already in memory, :code:`opencosmo` will use that data to construct the data returned by the second :code:`get_data` call. This is much faster than going to disk and reading the data a second time. However, it is not possible to avoid creating a copy in memory in this case. Halos above a mass of :math:`10^{14}` are randomly distributed throughout the data. Selecting only the rows with those high-mass halos requires creating new arrays in memory. 

The goal of :code:`opencosmo` is to allow you to spend less time worrying about data management and more time worrying about your science. The toolkit will handle moving data in and out of memory as efficiently as possible. You don't need to worry about the details.
