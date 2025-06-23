Working with Units
==================

In general, the raw output of cosmological simulations is *scale free*: coordinates and velocities use `comoving coordinates <https://en.wikipedia.org/wiki/Comoving_and_proper_distances#Comoving_distance_and_proper_distance>`_ and values are given in terms of the reduced Hubble constant *h*.

OpenCosmo allows you to convert between these conventions as needed based on what is most convinient for your analysis. The default convention is "comoving." This leaves coordinates and velocities in comoving units, but absorbs the *h* factor into the values of the data. The other available conventions are "physical", "scalefree" and "unitless." Note that "scalefree" and "unitless" will have the same numerical values.

You can readily convert between these conventions with the :py:meth:`opencosmo.Dataset.with_units` method:

.. code-block:: python

   import opencosmo as oc

   ds = oc.read("galaxyproperties.hdf5")
   ds = ds.with_units("physical")

You can also have work with a datasets in multiple unit conventions at the same time.

.. code-block:: python

   import opencosmo as oc

   # comoving is the default
   ds_comoving = oc.read("galaxyproperties.hdf5")
   ds_physical = ds_comoving.with_units("physical")
   # ds_comoving is not changed



When you filter a dataset with :py:meth:`opencosmo.Dataset.filter`, the filtering will be performed in the same unit convention as the dataset. If you provide a unitless value to a filter, it will be interpreted as a value with the same units as the column in question.

.. code-block:: python

   import opencosmo as oc

   ds = oc.read("haloproperties.hdf5").with_units("scalefree")
   ds_filtered = ds.filter(ds.col("fof_halo_mass") > 1e10)
   # This will filter out halos with mass greater than 1e10 * h^-1 Msun
   ds_comoving = ds.with_units("comoving")
   ds_filtered_comoving = ds_comoving.filter(ds_comoving.col("fof_halo_mass") > 1e10)
   # This will filter out halos with mass greater than 1e10 Msun (no factor of h!)

If you change unit conventions after performing a filter, the filter will still be applied in the original unit convention. For example:

.. code-block:: python

   import opencosmo as oc
   
   ds = oc.read("haloproperties.hdf5").with_units("scalefree")
   ds_filtered = ds.filter(ds.col("fof_halo_mass") > 1e10).with_units("comoving")
   print(ds_filtered.data["fof_halo_mass"].min())


should output a value of ~1.5 x 10^10 Msun, which is about 1e10/0.67.

   
