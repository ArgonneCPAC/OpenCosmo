opencosmo 1.3.1 (2026-05-27)
============================

Bugfixes
--------

- Requesting more rows than exist via :py:meth:`take <opencosmo.Dataset.take>` or :py:meth:`take_range <opencosmo.Dataset.take_range>` no longer raises a ``ValueError``. Instead, all available rows are returned. This applies to :py:class:`Dataset <opencosmo.Dataset>`, :py:class:`Lightcone <opencosmo.Lightcone>`, and :py:class:`StructureCollection <opencosmo.StructureCollection>`. (#240)
- Fix a bug that could cause renamed columns to be instantiated without the correct units
- Fixed a bug that could cause column arithmetic to fail with scalars.
- HealpixMap now correctly unpacks data when there is only a single map, instead of returning a dictionary. Mirrors
  behavior in Dataset etc.


New Features
------------

- :py:meth:`take <opencosmo.Dataset.take>`, :py:meth:`take_range <opencosmo.Dataset.take_range>`, and their equivalents on :py:class:`Lightcone <opencosmo.Lightcone>` and :py:class:`StructureCollection <opencosmo.StructureCollection>` now accept a ``mode`` keyword argument. Setting ``mode="global"`` when running under MPI causes ``n`` (or ``start``/``end``) to be interpreted across all ranks combined rather than per-rank. When the dataset is sorted, ranks coordinate to select from the globally-sorted order, so ``ds.sort_by("fof_halo_mass").take(1000, mode="global")`` returns exactly the 1000 most massive halos distributed across all ranks. (#240)
- :py:meth:`Dataset.filter <opencosmo.Dataset.filter>` now accepts masks created from expressions built from column arithmetic.
- :py:meth:`get_data <opencosmo.Dataset.get_data>` now supports :code:`jax` as an output format.
- Add the :py:meth:`Lightcone.get_pixels() <opencosmo.Lightcone.get_pixels()` to retrieve the healpix pixels covered by a lightcone.
- Add the :py:meth:`Lightcone.query_pixels <opencosmo.Lightcone.query_pixels>` method to query a lightcone based on healpix pixels.
- Added animate_halos function, which calls either "visualize_halo" or "halo_projection_array" in a loop to create an animated visualization of a given halo or set of halos.
- All :code:`evaluate` methods (e.g. :py:meth:`Dataset.evaluate <opencosmo.Dataset.evaluate>`) now support passing data to the function in any format supported by :py:meth:`get_data <opencosmo.Dataset.get_data>`.
- Calls to :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>` and :py:meth:`evaluate <opencosmo.Dataset.evaluate>` now accept an `allow_overwrite` flag. In this way you can "transform" a column by creating a derived column that depends on a column of the same name. The input will be the original column, and the output will be the new version.
- Diffsky catalog can be forced to keep groups during filtering etc. by setting the `get_top_host = True` flag when
  opening.
- The :py:class:`StructureCollection <opencosmo.StructureCollection>` now has a more complete string representation
- The :py:class:`StructureCollection <opencosmo.StructureCollection>` object now supports multi-step lightcones.


Improvements
------------

- Conversion to healsparse in :py:meth:`HealpixMap.get_data <opencosmo.HealpixMap.get_data` with `format = healsparse` has been rewritten, improving performance by roughly a factor of 5.
- Dataset instantiation and backend process has been reworked to allow for dynamic column updating.
- Difficult indexing routines have been rewritten in native code, allowing the removal of numba as a dependency.
- Improved spatial querying on lightcones, which should result in significant speedup for larger regions.
- The OpenCosmo internals now include a basic plugin architecture, which allows for specific data types to introduce
  modified behavior at particular points. Currently being used to dynamically re-build the `top_host_idx` column in
  diffsky data.
- You can now request HealpixMaps in Healpix format even when the map does not cover the full sky. Maps requested in this format will be returned as masked numpy arrays.


Miscellaneous
-------------

- The DatasetState object has been broken up to allow for more flexibility in instantiating datasets, laying the groundwork for future optimizations.



