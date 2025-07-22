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


