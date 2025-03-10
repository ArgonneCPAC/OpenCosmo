Working with MPI
================

OpenCosmo can read and write data inside an MPI environment. In general the API works exactly the same within an MPI context as it does otherwise, but there are some things to be aware of (see below).

How OpenCosmo Thinks About MPI
-------------------------------
The OpenCosmo MPI routines are primarily for filtering, transforming, and very large datasets. It should not generally be used for plotting or other end-product analysis steps. If you have a dataset that is larger than the memory available on your machine, you should consider using :py:func `opencosmo.open` and using :py:meth `opencosmo.Dataset.filter` and `opencosmo.Dataset.select` to reduce the data to a manageble size before it is loaded into memory.

OpenCosmo will automatically detect if it is operating within an MPI process. Each rank will be responsible for an equally-sized chunk of rows for all operations. Data can be filtered, transformed, and written exactly as it would be in a single-threaded context. Howeverer, there are some caveats:


I/O with Parallel HDF5
----------------------
Reading HDF5 data in parallel requires no additional work on your part. However parallel writes require that you have a copy of HDF5 on your system that has been compiled with parallel write support turned on. h5py does *not* come with this version of HDF5 by default. To install it, you will need to uninstall the version of h5py that is installed in your Python environment and manually re-install with MPI support enabled. As long as you have an appropriate C compiler on your system this should be straightforward. See `this entry in the h5py documentation https://docs.h5py.org/en/latest/build.html#custom-installation`_ for more details. 

Transformations Must be Consistent Across Ranks
------------------------------------------------

The OpenCosmo MPI routines should only be used to perform operations that would take prohibitively long on a single thread. You should always perform the same operations on your dataset on all ranks, and then write the output to a new file. If you perform different operations on the same dataset the results may still write but may not be what you expect.

"Take" Operations
-----------------

Because OpenCosmo works with chunks of rows, :py:meth: `opencosmo.Dataset.take` can result in some ranks returning no data. For example, if you open a dataset with 1000 rows on 4 ranks each rank will be responsible for 250 rows. If you then take 300 rows from the dataset, ranks 0 and 1 will return 250 and 50 rows respectively, while ranks 2 and 3 will return no data. This is the expected behavior, but ranks 2 and 3 will throw an exception. You should wrap any calls to :py:meth: `opencosmo.Dataset.take` in a try/except block to handle these exceptions. This is generally not a problem if you are using take with at="random".

When performing a take operation, OpenCosmo will verify the take operation is the same across all ranks. If it is not, the input from rank 0 will be used and a warning will be printed.

