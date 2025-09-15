Opencosmo 0.9.0 (2025-09-15)
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


Opencosmo 0.8.0 (2025-07-22)
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


