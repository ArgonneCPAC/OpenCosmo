:py:meth:`Dataset.select <opencosmo.Dataset.select>`, :py:meth:`Dataset.with_new_columns <opencosmo.Dataset.with_new_columns>`, and :py:meth:`Dataset.filter <opencosmo.Dataset.filter>` (and the equivalents on :py:class:`Lightcone <opencosmo.Lightcone>` and :py:class:`StructureCollection <opencosmo.StructureCollection>`) now accept a ``mode`` keyword argument. The default ``mode="global"`` combines scalar reductions across all ranks under MPI before they are used, so every rank ends up with the same value. Pass ``mode="local"`` to restrict the reduction to each rank's own chunk. This applies to top-level scalar selections, scalar reductions nested inside derived column expressions, and scalars used in filter masks:

.. code-block:: python

   m = oc.col("fof_halo_mass")

   # Scalar selection — defaults to the cross-rank global value
   global_min = ds.select(min_mass=m.min()).get_data()

   # Per-rank scalar
   local_min = ds.select(min_mass=m.min(), mode="local").get_data()

   # Derived column normalized against the global mean and std
   ds = ds.with_new_columns(zscore=(m - m.mean()) / m.std())

   # Filter against a globally-computed threshold
   ds = ds.filter(m > m.mean())

``mode`` has no effect on plain column selections, on expressions without scalar reductions, or when not running under MPI.
