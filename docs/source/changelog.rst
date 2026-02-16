opencosmo 1.1.1 (2026-02-16)
===============================

Documentation
-------------

- Regularize some issues in the documentation, provide some additional information in the getting started doc.


Deprecations and Removals
-------------------------

- Retrieving data with the :py:meth:`.data <opencosmo.Dataset.data>` attribute has been marked as deprecated.


opencosmo 1.1.0 (2026-02-10)
===============================

New Features
------------

- Add a new :meth:`reduce <opencosmo.analysis.reduce>` function, which allows results from multiple processes to be combined into a single result by summing, multiplying, or averaging.
- Lightcones now correctly carry a :py:class:`HealpixRegion <opencosmo.spatial.HealpixRegion>` with the pixels that
  contain their data.
- :meth:`Dataset.evaluate <opencosmo.Dataset.evaluate>`, :meth:`Lightcone.evaluate <opencosmo.Lightcone.evaluate>` and :meth:`StructureCollection.evaluate_on_dataset <opencosmo.StructureCollection.evaluate_on_dataset>` now support a :code:`batch_size` argument, which allows you to specify the number of rows that should computed at once.


Improvements
------------

- Healpix Regions now print nicely
- Healpix maps no longer need to store their pixel numbers in a file when they cover the full sky.
- The OpenCosmo Header can now store large arrays.


opencosmo 1.0.4 (2026-02-04)
===============================

Bugfixes
--------

- Dictionary kwargs with the same name as the datasets in a lightcone are now correctly mapped
- Fix a bug that could prevent unitless columns to be converted to numpy arrays when `format = "numpy"` is set
- Fix a bug that that could cause data used for verification of evaluated columns to carry units even if the evaluated column requested data without units


opencosmo 1.0.3 (2026-01-30)
===============================

Bugfixes
--------

- Fix a bug that could cause MPI writes to fail because dynamically-created communicators were not cleaned up correctly.


opencosmo 1.0.2 (2026-01-28)
===============================

Bugfixes
--------

- Fix a bug that could cause a bad error message when a dependnecy cycle is detected in derived columns.
- Fix a bug that could cause derived column instantiation to fail if other derived columns with specific properties were not included in the selection.


opencosmo 1.0.1 (2026-01-21)
===============================

Improvements
------------

- Disable unneeded verification step for eagerly-evaluate columns in :py:meth:`Dataset.evaluate <opencosmo.Dataset.evaluate>`.


opencosmo 1.0.0 (2026-01-21)
===============================

Bugfixes
--------

- Fix a bug that caused structure collections to not open correctly if the individual datasets were lightcone datasets. (#103)
- Fix a bug that prevented adding together columns with logarithmic units (#142)
- Attempting to sort by a column that is not in the dataset now errors correctly.
- Correctly update order across ranks when stacking lightcones.
- Fix a bug that could cause :py:meth:`StructureCollection.evaluate <opencosmo.StructureCollection.evaluate>` to fail if the :code:`dataset` argument is set to halo_properties or galaxy_properties.
- Fix a bug that could cause created columns to be evaluated twice instead of correctly being cached.
- Fix a bug that could cause filtering to fail on very short datasets.
- Fix a bug that could cause lightcone stacking to fail with large numbers of datasets
- Fix a bug that could cause segfaults (in numba-accelerated code) when running take operations
- Fix a bug that could cause unit conversions on structure collections to not propogate to a halo_properties or galaxy_properties dataset.
- Fix a bug that could cause user-created columns to be ordered incorrectly if they were inserted into a sorted dataset or lighcone.
- Fix a bug that could cause writes to fail for a large SimulationCollection
- Fix a bug that could cause writing to segfault when working in MPI
- Submitting a filter without actually passing filters will now return the original dataset rather than an empty dataset.


New Features
------------

- :code:`with_units` can now be used to provide unit conversions, in addition to changing conventions (#43)
- Descriptions of columns can now be accessed with :py:class:`Dataset.descriptions <opencosmo.Dataset.descriptions>` (#122)
- :code:`with_new_columns` now accepts a :code:`descriptions` argument for providing column descriptions
- :py:meth:`StructureCollection.evaluate <opencosmo.StructureCollection.evaluate>` now performs evaluation on individual structures when `dataset` argument is passed.
- :py:meth:`opencosmo.write` now works in MPI contexts even if parallel-hdf5 is not installed.
- Add :py:meth:`StructureCollection.evaluate_on_dataset <opencosmo.StructureCollection.evaluate_on_dataset>`, which performs evaluate on a single dataset in the structure collection without chunking.
- Add a number of common column combinations that can be used to add columns to a dataset with
  :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns`. See the :ref:`column API reference <Provided Column Combinations>` for details
- Added additional supported outpt formats "pandas", "polars" and "arrow"
- Added support for working with maps stored with a Healpix decomposition with new the :py:class:`opencosmo.HealpixMap` class.
- Column filters can now be combined with boolean operators & (and) and | (or).
- Columns created with :py:meth:`opencosmo.col` now support :code:`.log10()`, :code:`.exp10()` and :code:`.sqrt()`
- Columns that are read from disk are now cached, ensuring that requesting data second time is always fast.
- When writing lightcones, datasets from adjacent redshift slices will now be stacked into a single dataset if their combined length is small enough.


Improvements
------------

- :py:meth:`StructureCollection.select <opencosmo.StructureCollection.select>` and :py:meth:`StructureCollection.drop <opencosmo.StructureCollection.drop>` now follow the same semantics as :py:meth:`StructureCollection.evaluate <opencosmo.StructureCollection.evaluate>` for passing columns from multiple datasets in a single function call.
- :py:meth:`StructureCollection.select <opencosmo.StructureCollection.select>`, :py:meth:`StructureCollection.drop <opencosmo.StructureCollection.drop>`, and :py:meth:`StructureCollection.evaluate <opencosmo.StructureCollection.evaluate>` now support specifying columns in nested collections.
- Add the ability to create datasets entirely in memory, which is used at present to support downgrading healpix maps.
- Evaluations in individual dataset are now performed lazily unless :code:`insert = False`
- Opening multiple files will now fail with a clear error message if one or more of the files is not opencosmo-formatted.
- Reads from hdf5 now use :code:`read_direct`, which improves read performance especially with large datasets.
- Rewrite DataIndex to be fully functional, and accelerate with Numba.
- The logic behind :py:meth:`oc.write <opencosmo.write>` has been completely rewritten to be more functional, reliable, and easy to extend. This change does not affect how the function is used, and does not make any changes to the data format.
- Writing in an MPI context will now fail with an error if no rank has data to write
- :py:meth:`Dataset.evaluate <opencosmo.Dataset.evaluate>` is now performed lazily when :code:`insert = True`.


Miscellaneous
-------------

- Column management has been reworked and centralized
- Move all annotation-only imports behind a :code:`TYPE_CHECKING` block and add :code:`from __future__ import annotations` to most files. This significantly improves initial import time.
- Unit handling has been fully rewritten, making it much more flexible for backend work.


Deprecations and Removals
-------------------------

- Functions passed into :py:meth:`Dataset.evaluate <opencosmo.Dataset.evaluate>` must now always explicitly list columns as arguments"
- opencosmo.read (deprecated since 0.7) has been removed.
- opencosmo.open_linked_files (deprecated sinced 0.8) has been removed
- opencosmo.Dataset.collect has been removed


opencosmo 0.9.6 (2025-10-20)
============================

Bugfixes
--------

- Make diffstar pop optional in diffsky parameters

opencosmo 0.9.4 (2025-09-26)
============================

Bugfixes
--------

- Fix a bug that could cause opening halos and galaxies only to fail


opencosmo 0.9.3 (2025-09-25)
============================

Bugfixes
--------

- Fix an issue that could cause opens with multiple files to fail in MPI contexts


opencosmo 0.9.2 (2025-09-25)
============================

Bugfixes
--------

- Fix a but that could cause opening properties and profiles without particles to fail.


opencosmo 0.9.1 (2025-09-16)
============================

Bugfixes
--------

- Re-add license file to package for conda-forge compatability


opencosmo 0.9.0 (2025-09-15)
============================

Features
--------

- Unitful values in headers now carry astropy units and transform under unit transformations (#70)
- :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>` can now take numpy arrays or astropy quantities (#77)
- Add a new "evaluate" method to the standard API, which fills the role of "with_new_columns" for cases when the computation is more than just a simple algebraic combination of columns (#77)
- Add "sort_by" to the standard API, which allows ordering by the value of a column. (#112)
- Mutli-dimensional columns are now unpacked correctly when only reading a single row
- You can now drop entire datasets from a StructureCollection using :py:meth:`with_datasets <opencosmo.StructureCollection.with_datasets>`
- You can now dump datasets and some data from collections to parquet with :py:meth:`opencosmo.io.write_parquet`
- You can now remove datasets from a StructureCollection with :code:`with_datasets`


Bugfixes
--------

- Fixed a bug that could cause MPI writes to stall after queries that returned small numbers of rows. (#99)
- Fixed a bug that could cause printing lightcone summary to fail when column originally contained "redshift" column that had been dropped. (#101)
- Fix a bug for installing analysis tools


Deprecations and Removals
-------------------------

- StructureCollections now always require a dataset be specified when calling :py:meth:`select <opencosmo.StructureCollection.collect>`


Misc
----

- #106
- Dependency management now handled by UV
- Minimum required version of mpi4py has been decreased to match version available in ALCF base python environments.
- Parameters accessed through a dataset are now returned as dictionaries
- Remove significant amounts of dead code, reorganize for clarity
- Replace astropy.table.Table with astropy.table.QTable, which generally handles units much more cleanly
- Update evaluate to avoid evaluating first row twice
- Update evaluate to respect default parameters


opencosmo 0.8.0 (2025-07-22)
============================

Features
--------

- StructureCollections now filter out structures with no particles by default (#75)
- Columns can now be dropped with `Dataset.drop <opencosmo.Dataset.drop>` (inverse of "select") (#77)
- Data can now be requested from datasets as Numpy arrays (#78)
- Data retrieved with the :code:`.data` attribute on datasets are now cached (#78)
- Added a new "Lightcone" collection type.
- Analysis modules can now be installed with new command line script :code:`opencosmo install`
- Data producers can now define user-provided flags that determine which datasets should be loaded in a multiple-dataset or group of files.
- The opencosmo.analysis module now includes tools for creating visualizations of halos


Bugfixes
--------

- StructureCollection and Datasets no longer raises StopIteration error if empty (prints warning) (#84)
- Fixed yt interface for gravity-only simulations (#90)


Improved Documentation
----------------------

- Add towncrier for automated changelog management


Deprecations and Removals
-------------------------

- open_linked_files has been depcrecated and will be removed in the future. Use :py:meth:`opencosmo.open` instead.


Misc
----

- Add installation from files with multiple datasets
- Data opening logic has been rewritten from scratch, singnificantly improving performance when opening many file.
- Partitioning in MPI now ignores regions that do not have data
- The header reading logic has been generalized to allow more flexibility in defining new data types
- Unit handling now supports data stored in conventions other than scalefree


