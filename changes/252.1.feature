:py:meth:`Dataset.select <opencosmo.Dataset.select>` can now retrieve scalar summary statistics directly. Pass scalar expressions as keyword arguments — ``get_data()`` returns an astropy Quantity for a single scalar or a dict of Quantities for multiple:

.. code-block:: python

   min_mass = ds.select(min_mass=oc.col("fof_halo_mass").min()).get_data()

   stats = ds.select(
       min_mass=oc.col("fof_halo_mass").min(),
       max_mass=oc.col("fof_halo_mass").max(),
   ).get_data()

Scalar reductions respect any prior ``filter()`` or ``bound()`` calls. Scalar and column selections cannot be mixed in a single ``select()`` call.
