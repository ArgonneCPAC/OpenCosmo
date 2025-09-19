Working with Units
==================


The raw output of cosmological simulations produced with HACC (and many other cosmology codes) is in *scale free* units: coordinates and velocities use `comoving coordinates <https://en.wikipedia.org/wiki/Comoving_and_proper_distances#Comoving_distance_and_proper_distance>`_ and values such as mass include terms of the reduced Hubble constant *h*. Some downstream data products, such as synthetic galaxy catalogs, may be in other conventions.

OpenCosmo allows you to convert between these conventions as needed based on what is most convinient for your analysis. The default convention is "comoving," which leaves coordinates and velocities in comoving units but absorbs any factors of h into their values. The other available conventions are "physical", "scalefree" and "unitless." 

The "comoving", "physical" and "unitless" conventions will always be available, while the "scalefree" convention will only be available if the raw data is already stored in this convention. The "unitless" convention will always return the raw data in the file, regardless of the convention the data is stored in.


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

Converting Columns to an Equivalent Unit
----------------------------------------

You can convert all columns with a given unit into a different unit with the :code:`conversions` argument:


.. code-block:: python

   import opencosmo as oc
   
   ds = oc.read("haloproperties.hdf5")
   conversions = {u.Mpc: u.lyr}
   ds = ds.with_units(conversions=conversions)

In the new dataset, all columns that originally had units of megaparsecs will be converted to lightyears. All-column conversions are always peformed after a change of unit conventions. Changing units *after* doing a conversion always clears the conversions.

.. code-block:: python

   import opencosmo as oc
   
   ds = oc.read("haloproperties.hdf5")
   conversions = {u.Mpc: u.lyr}
   ds = ds.with_units(conversions=conversions)



Single-Column Conversions
-------------------------

You can also use :code:`with_units` to convert the values in individual columns to their values in an equivalent unit:

.. code-block:: python

   import astropy.units as u

   dataset = oc.read("haloproperties.hdf5").with_units(
        fof_halo_center_x = u.lyr,
        fof_halo_center_y = u.lyr,
        fof_halo_center_z = u.lyr,
   )

Unit conversions like these are always performed *after* any change in unit convention, and changing unit conventions clears any existing unit conversions:

.. code-block:: python

    # this works
    dataset = dataset.with_units(fof_halo_mass=u.kg)

    # this clears the previous conversion,
    # the masses are now in Msun / h
    dataset = dataset.with_units("scalefree")

    # This now fails, because the units of masses
    # are Msun / h, which cannot be converted to kg
    dataset = dataset.with_units(fof_halo_mass=u.kg)

    # this will work, the units of halo mass in the "physical"
    # convention are Msun (no h), and the change of convention
    # happens before the conversions
    dataset = dataset.with_units("physical", fof_halo_mass=u.kg, fof_halo_center_x=u.lyr)

    # reset all units
    dataset = dataset.with_units("physical")


Unit conversions on :py:class:`Lightcones <opencosmo.Lightcone>` and :py:class:`SimulationCollections <opencosmo.SimulationCollection>` behave identically to single datasets. In :py:class:`StructureCollections <opencosmo.StructureCollections>`, unit conversions must be passed on a per-dataset basis:

.. code-block:: python

   import astropy.units as u

   structures = oc.open("haloproperties.hdf5", "haloparticles.hdf5")
   structures = structures.with_units(
        halo_properties={"fof_halo_mass": u.kg},
        dm_particles={"mass": u.kg}
   )

Conversion Precedence
---------------------

In cases where a blanket conversion is provided alongside a conversion for a specific column, the specific conversion always take precedence:

.. code-block:: python

   import astropy.units as u

   conversions = {u.Mpc: u.lyr}
   ds = ds.with_units(conversions=conversions, fof_halo_center_x=u.km)

All columns with units of megaparsecs will be converted to lightyears, except for the :code:`fof_halo_center_x` column which will be converted to kilometers.

Structure Collection Conversions
--------------------------------

When working with a structure collection, you can provide conversions that apply to the entire collection, as single dataset inside the collection, or individual columns within a given dataset. As you might expect, conversions on an individual dataset takes precedence over those that apply to all datasets.

.. code-block:: python

            import astropy.units as u

            conversions = {u.Mpc: u.lyr}
            structures = structures.with_units(
                conversions=conversions
                halo_properties = {
                    "conversions": {u.Mpc: u.km},
                    "fof_halo_center_x": u.m
                }
            )

In this example, all values in Mpc will be converted to lightyears, except in the "halo_properties" dataset, where they will be converted to kilometers. The column "fof_halo_center_x" in "halo_properties" will be converted to meters instead.

Clearing Conversions
--------------------

Conversions are always cleared when changing unit conventions, or you can also clear them by calling :code:`with_units` with no arguments.

.. code-block:: python

   dataset = oc.read("haloproperties.hdf5").with_units(
        conversions={u.Mpc: u.lyr},
        fof_halo_center_x = u.lyr,
   )

   dataset = dataset.with_units()
   # all unit conversion reset


