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

The `open` function returns a `Dataset` object, which holds the raw data as well as information about the simulation. You can easily access the data and cosmology directly as Astropy objects:

.. code-block:: python

   dataset.data
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

