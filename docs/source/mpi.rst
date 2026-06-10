Working with MPI
================

OpenCosmo can read and write data in an MPI environment. In general the API works exactly the same within an MPI context as it does otherwise, but there are some things to be aware of in the current version of the library (see below). More flexibility in working in an MPI context is planned for future work.

In general, scripts that work in a non-MPI context should also work in an MPI context without any modification. The toolkit will automatically handle chunking the dataset across ranks, and coordinating to write data if necessary.

Check if MPI is Active
----------------------
You can check if your script is running with MPI using the :py:meth:`has_mpi <opencosmo.mpi.has_mpi>` convinience method:

.. code-block:: python

   from opencosmo.mpi import has_mpi
   print(has_mpi()) # True or False


Reading and Writing Data with MPI
---------------------------------
When you :py:meth:`open <opencosmo.open>` data with MPI, :code:`opencosmo` automatically chunks the data across all processes. Each process recieves a roughly-equal-sized chunk of data. The chunking is done spatially, meaning each rank's data will be fall inside some contiuous spatial region.

Once you have opened data, the APIs are the same as if you were operating without MPI. All method calls operate solely on the local processes data without any communication with the other processes. This is well suited for distributing analyses across large-scale datasets when they do not involve spatial work. Additional coordination tools for spatial analyses are planned for a future release.

You do not need to do anything special to write data in parallel. Simply call :py:meth:`oc.write <opencosmo.write>` from all processes at the same time. :code:`opencosmo` will automatically coordinate between processes to write your file. However when working with large datasets and/or many MPI ranks, we strongly recommend installing a copy of HDF5 with parallel support. Parallel hdf5 allows multiple ranks to write data simultaneously, which will significantly decrease the amount of time required to write the data. See :doc:`installation` for details on how to install a parallel version of hdf5 on your system.

Combining Results Across Processes with :code:`reduce`:
-------------------------------------------------------
:code:`opencosmo` contains convinience functions for combining the results of a computation across ranks. The :meth:`reduce <opencosmo.analysis.reduce>` function allows you to sum, multiply, or average results from several different processes into a single result. For example, suppose you are working with a very large simulation using MPI and you want to compute the halo mass function across the entire simulation:

.. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        import opencosmo as oc
        from opencosmo.analysis import reduce

        ds = oc.open("haloproperties.hdf5")

        def halo_mass_function(fof_halo_mass, log_bins, box_size):
            log_mass = np.log10(fof_halo_mass)
            hist, _ = np.histogram(log_mass, log_bins)
            return hist / np.diff(log_mass) / box_size ** 3

        bins = np.linspace(10, 15)
        box_size = ds.header.simulation["box_size"].value

        results = reduce(ds, halo_mass_function, log_bins = bins, box_size = box_size, vectorize=True)
        if histogram is not None:
            plt.plot(bins, results["halo_mass_function"])
            plt.savefig("hmf.png")

:meth:`reduce <opencosmo.analysis.reduce>` uses :meth:`evalute <opencosmo.Dataset.evaluate>` to perform its computation. As a result, the expected signature of the computation function is identical. Any additional keyword arguments are passed directly to the underlying :code:`evaluate` implementation, with the exception of :code:`insert` which is ignored. Note this does mean that the exact set of expected arguments will depend on the type of the dataset you are computing with. See the API reference for the various :code:`opencosmo` dataset and collection types for more details.

Additional convinience functions for working with MPI are planned for future releases.

Important Caveats
-----------------


"Take" and "Select" Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a dataset is opened in an MPI context, the data is chunked across all ranks. :py:meth:`opencosmo.Dataset.take` operations will always operate on the data that is local to the given rank. For example, taking 100 rows at random on all ranks will actually take 100*N_ranks rows, distributed evenly across the ranks. Taking 100 rows with :code:`at = "start"` will take the first 100 rows on each rank.

When calling :py:meth:`select <opencosmo.Dataset.select>` or :py:meth:`drop <opencosmo.Dataset.drop>`, it is important to be sure to always include the same columns on all ranks if you intend to write data. If you attempt to write data and some processes have different columns than others, the write will fail.

When retrieving scalar values via :py:meth:`select <opencosmo.Dataset.select>` in an MPI context, scalar reductions operate on the local chunk by default. To compute a scalar across all ranks, pass :code:`mode="global"` — see below.

Spatial Queries
~~~~~~~~~~~~~~~
In OpenCosmo, raw data is ordered according to its location in the spatial index. When a dataset is loaded with MPI, each rank recieves an equal share of the regions in the spatial index. As a result, most spatial queries are likely to return no data for most ranks. Ranks that fall completely outside the query region will return a zero-length dataset. :meth:`opencosmo.write` will handle the zero-length datasets automatically.

You can retrieve the region the local dataset is contained with in by calling :meth:`dataset.region <opencosmo.Dataset.region>`. One possible workflow is to perform different spatial queries for each rank depending on the region that is local to that rank.

Currently OpenCosmo does not support sharing data across ranks, such as when a given spatial query crosses a rank boundary. This will be improved in the future.

Global Scalar Reductions
-------------------------

When using :py:meth:`select <opencosmo.Dataset.select>` to compute scalar summary statistics (e.g. ``.min()``, ``.mean()``), the default behavior is :code:`mode="local"`, meaning each rank computes the reduction over only its own chunk of data. This is correct when you want per-rank statistics or when you are aggregating results yourself.

To compute a scalar that reflects the **entire dataset across all ranks**, pass :code:`mode="global"`:

.. code-block:: python

   import opencosmo as oc

   ds = oc.open("haloproperties.hdf5")

   # Each rank returns its local minimum independently
   local_min = ds.select(min_mass=oc.col("fof_halo_mass").min()).get_data()

   # All ranks receive the global minimum across the full dataset
   global_min = ds.select(
       min_mass=oc.col("fof_halo_mass").min(),
       mode="global",
   ).get_data()

:code:`mode="global"` has no effect when not running under MPI, or when the selection contains no scalar reductions.

Note that for the operationss ``std``, ``var``, ``median``, and ``quantile``, all per-rank data is gathered to rank 0, the reduction is computed there, and the result is broadcast back. For very large datasets this may be memory-intensive on rank 0.

Note that :code:`mode="global"` also works in combination with scalar arithmetic:

.. code-block:: python

   m = oc.col("fof_halo_mass")
   # Normalize each rank's data using the global mean and std
   ds = ds.select("*", zscore=(m - m.mean()) / m.std(), mode="global")


