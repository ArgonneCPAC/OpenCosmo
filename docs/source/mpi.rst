Working with MPI
================

OpenCosmo can read and write data in an MPI environment. In general the API works exactly the same within an MPI context as it does otherwise, but there are some things to be aware of in the current version of the library (see below). More flexibility in working in an MPI context is planned for the next version of the library.

I/O with Parallel HDF5 and Select Operations
--------------------------------------------
Reading HDF5 data in parallel requires no additional work on your part. However parallel writes require that you have a copy of HDF5 on your system that has been compiled with parallel write support turned on. See :doc:`installation` for details on how to install a parallel version of hdf5 on your system.

Currently, OpenCosmo does not support writing data in an MPI context unless all ranks are writing to the same data. In practices this means that all ranks must have the same columns in their data, so all :py:meth:`opencosmo.Dataset.select` operations must be identical across ranks. 


"Take" Operations
-----------------

When a dataset is opened in an MPI context, the data is chunked across all ranks. :py:meth:`opencosmo.Dataset.take` operations will always operate on the data that is local to the given rank. For example, taking 100 rows at random on all ranks will actually take 100*N_ranks rows, distributed evenly across the ranks. Taking 100 rows with :code:`at = "start"` will take the first 100 rows on each rank.

Spatial Queries
---------------
In OpenCosmo, raw data is ordered according to its location in the spatial index. When a dataset is loaded with MPI, each rank recieves an equal share of the regions in the spatial index. As a result, most spatial queries are likely to return no data for most ranks. Ranks that fall completely outside the query region will return a zero-length dataset. There are three primary options for handling spatial queries in an MPI context.

**Collect**

Calling :meth:`opencosmo.Dataset.collect` will






