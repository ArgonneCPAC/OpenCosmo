Working with MPI
================

OpenCosmo can read and write data in an MPI environment. In general the API works exactly the same within an MPI context as it does otherwise, but there are some things to be aware of in the current version of the library (see below). More flexibility in working in an MPI context is planned for the next version of the library.

File Opening Modes
------------------

:py:meth:`opencosmo.open` accepts a :code:`mpi_mode` argument which can be used to control the opening behavior under mpi. The currently supported modes are:

- :code:`"ordered"` (default): Each rank recieves a continuous chunk of data (corresponding to a specific spatial region). The chunks are ordered by rank number
- :code:`None`: No chunking is performed, and all ranks have access to the full dataset.

I/O with Parallel HDF5
----------------------
Reading HDF5 data in parallel requires no additional work on your part. However parallel writes require that you have a copy of HDF5 on your system that has been compiled with parallel write support turned on. See :doc:`installation` for details on how to install a parallel version of hdf5 on your system.


"Take" Operations
-----------------

When a dataset is opened in an MPI context, the data is chunked across all ranks. :py:meth:`opencosmo.Dataset.take` operations will always operate on the data that is local to the given rank. For example, taking 100 rows at random on all ranks will actually take 100*N_ranks rows, distributed evenly across the ranks. Taking 100 rows with :code:`at = "start"` will take the first 100 rows on each rank.

Spatial Queries
---------------
In OpenCosmo, raw data is ordered according to its location in the spatial index. When a dataset is loaded with MPI, each rank recieves an equal share of the regions in the spatial index. As a result, most spatial queries are likely to return no data for most ranks. Ranks that fall completely outside the query region will return a zero-length dataset. :meth:`opencosmo.write` will handle the zero-length datasets automatically.

You can retrieve the region the local dataset is contained with in by calling :meth:`opencosmo.Dataset.region`. One possible workflow is to perform different spatial queries for each rank depending on the region that is local to that rank.

Currently OpenCosmo does not support sharing data across ranks, such as when a given spatial query crosses a rank boundary. This will be improved in the future.

