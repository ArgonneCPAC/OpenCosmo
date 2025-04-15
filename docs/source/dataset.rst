Working with OpenCosmo Datasets
===============================

The Dataset is the core construct in OpenCosmo. It provides convineience functions for manipulating simulation data with full awareness of the cosmological context.

Reading a Dataset
------------------

Datasets can be created with :py:func:`opencosmo.read` or :py:func:`opencosmo.open`. Use read if you're working with a small amount of data, and it's not a problem to load it all into memory at once. Use open if the dataset is large and you plan to filter it down. For more details, see :doc:`io`.

Working with Datasets
---------------------

You can always retrieve the data and cosmology from a dataset with the :py:attr:`data` and :py:attr:`cosmology` attributes, respectively

.. code-block:: python

   import opencosmo as oc

   ds = oc.read("galaxyproperties.hdf5")
   data = ds.data
   print(ds.data)
   print(ds.cosmology)

**Output:**

.. code-block:: text

   <Table length=51282>
   block fof_halo_center_x fof_halo_center_y  ...  gal_zmet_star unique_tag
                Mpc          3     Mpc         ...                     
   int32      float32           float32       ...   float32      int64
   ----- ----------------- -----------------  ...  ------------- ----------
       0         1.4922178         18.814196  ..   0.012433555      52332

   # many more rows...

   FlatLambdaCDM(name=None, H0=<Quantity 67.66 km / (Mpc s)>, Om0=0.3096446816186967, Tcmb0=<Quantity 0. K>, Neff=3.04, m_nu=None, Ob0=0.04897468161869667)

OpenCosmo provides you with tools to filter and transform the data in its full cosmological context, then save the results to a file you can use later. For example, suppose we only want to work with galaxies with a stellar mass greater than 10^10 solar masses. We can filter the dataset like this:

.. code-block:: python

   import opencosmo as oc

   ds = oc.read("galaxyproperties.hdf5")
   ds = ds.filter(os.col("gal_mass_star") > 1e10)
   print(ds)


**Output:**

.. code-block:: text

   OpenCosmo Dataset (length=1004)
   Cosmology: FlatLambdaCDM(name=None, H0=<Quantity 67.66 km / (Mpc s)>, Om0=0.3096446816186967, Tcmb0=<Quantity 0. K>, Neff=3.04, m_nu=None, Ob0=0.04897468161869667)
   First 10 rows:
   block fof_halo_center_x fof_halo_center_y fof_halo_center_z ... gal_zmet_gas gal_zmet_sfgas gal_zmet_star unique_tag
                Mpc               Mpc               Mpc        ...                                              Gyr
   int32      float32           float32           float32      ...   float32       float32        float32      int64
   ----- ----------------- ----------------- ----------------- ... ------------ -------------- ------------- ----------
       1         37.519623         22.583586         13.334438 ...  0.013368793         -101.0   0.025042534   36743270
       1          41.12425         19.333538         22.081404 ...   0.01560147         -101.0   0.026290782   45142210
       1         44.857594          20.63389         13.371356 ...   0.03799247         -101.0   0.027433265   46191772
       1         44.857594          20.63389         13.371356 ... 0.0050295726         -101.0    0.02457484   46191772
       1         44.857594          20.63389         13.371356 ...  0.009271776         -101.0   0.027204797   46191772
       1         46.449963         22.552586         7.9736605 ...  0.024930356         -101.0    0.02424204   52461622
       1         46.449963         22.552586         7.9736605 ...  0.027771743         -101.0     0.0278134   52461622
       3         46.459282         27.993248          5.871807 ... 0.0047545577         -101.0   0.025008023   54061116
       5         41.804413         14.022727          34.11428 ...   0.02313051         -101.0   0.027187062   40387798
       5          43.33499         12.274037         39.537704 ... 0.0063861427         -101.0   0.026727734   50881802


Note that transforming a dataset always produces a new dataset. The original dataset will be left as-is. This allows us to perform many transformations in sequence:

.. code-block:: python

   import opencosmo as oc

   ds = oc.read("galaxyproperties.hdf5")
   ds = ds.filter(os.col("gal_smass_star") > 1e11)
          .take(100, at="random")
          .with_units("physical")
   print(ds)

**Output:**

.. code-block:: text

   OpenCosmo Dataset (length=100)
   Cosmology: FlatLambdaCDM(name=None, H0=<Quantity 67.66 km / (Mpc s)>, Om0=0.3096446816186967, Tcmb0=<Quantity 0. K>, Neff=3.04, m_nu=None, Ob0=0.04897468161869667)
   First 10 rows:
   block fof_halo_center_x fof_halo_center_y ... gal_zmet_sfgas gal_zmet_star unique_tag
           Mpc / littleh     Mpc / littleh   ...                                 Gyr
   int32      float32           float32      ...    float32        float32      int64
   ----- ----------------- ----------------- ... -------------- ------------- ----------
      11          52.17189         17.906551 ...         -101.0   0.024718575  111212606
      13         55.563084         11.117564 ...         -101.0   0.025339618  112265556
      19         27.842876          50.48995 ...         -101.0   0.025407294   35330058
      19         29.951859         53.379898 ...         -101.0   0.024370003   62594048
      24         33.342503         46.648335 ...         -101.0    0.02615105   66225282
      24         40.054035         40.401333 ...         -101.0   0.026113253   89292834
      26          43.80568         60.209282 ...         -101.0    0.02931065   96207982
      36           8.11038          8.850166 ...         -101.0   0.025345985   24690034
      36           9.66428         4.1855025 ...         -101.0   0.026431121   34102674
      50          9.484544         51.750984 ...         -101.0   0.024551544   39497092



 
