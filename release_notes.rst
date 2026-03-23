opencosmo 1.2.0 (2026-03-23)
============================

Bugfixes
--------

- Fix a bug that could cause evaluation verification to fail because of unit checks
- Fix a bug that could cause spatial queries to fail if there were no objects in the parts of the spatial index that
  partially intersect with the query region.
- Units are converted to comoving in :py:class:`halo_projection_array <opencosmo.analysis.halo_projection_array>`
  :py:meth:`create_yt_dataset <opencosmo.analysis.create_yt_dataset>` raises an error if the inputted dataset has factors of "littleh" in the units
  converting yt field keys to tuple in :py:class:`halo_projection_array <opencomo.analysis.halo_projection_array>`.


New Features
------------

- Add :py:meth:`box_search <opencosmo.Lightcone.box_search>` to the Lightcone API for performing
  spatial regions in a given RA and Dec range. (feature)
- - :py:class:`halo_projection_array <opencosmo.analysis.halo_projection_array>` visualizations now support empty panels.
  - Added ``cmap_norm`` and ``cmap_norms`` parameters to :py:class:`halo_projection_array <opencosmo.analysis.halo_projection_array>`
  - Changed the default text label color to `"lightgray"` (998e1c5)
- :py:meth:`select <opencosmo.Dataset.select>` and :py:meth:`drop <opencosmo.Dataset.drop>` now accept wildcards.
- :py:meth:`select <opencosmo.Dataset.select>` no longer requires column names to be passed as a list or tuple.
- :py:meth:`select <opencosmo.Dataset.select>` now support passing derived columns as keyhword arguments, which are instantiated as part of the selection.
- :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>` now accept simple columns created with :py:meth:`oc.col <opencosmo.col>`, effectively allowing for renaming.
- Datasets now support accessing column units with the :py:attr:`.units <opencosmo.Dataset.units>` attribute.
- Introduce a :py:func:`make_skybox <opencosmo.make_skybox>` function for creating boxes with RA and Dec ranges for
  performing spatial queries on lightcone data.
- Lightcones now automatically create a :code:`ra` and :code:`dec` column if they do not exist but can be derived from the existing columns.


Improvements
------------

- :py:meth:`StructureCollection.with_datasets <opencosmo.StructureCollection.with_datasets>` now accepts any iterable rather than just lists.
- Headers are now no longer as strict about what groups are required, assuring non-hacc datasets can be used without modifying the available parameter models.
- Python 3.14 is now tested in CI and fully supported
- The :py:meth:`open <opencosmo.open>` function has been rewritten to be more systematic, and significant internal documentation has been added to provide context around the decision making process.


Miscellaneous
-------------

- The :code:`output` parameter in :py:meth:`Dataset.get_data <opencosmo.Dataset.get_data>` has been renamed to :code:`format`. The method will continue to accept :code:`output` as a keyword argument for the time being, but will print a deprecation warning.


Deprecations and Removals
-------------------------

- This is the last release that will support Python 3.11
- hdf5plugin 5.1 is now explicitly excluded because of user reports of issues with blosc filters.



