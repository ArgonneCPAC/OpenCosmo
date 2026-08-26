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

Distributing Lightcones by Redshift
-----------------------------------
The default spatial chunking has every rank open every file. A lightcone is stored as one file per redshift step, and a lightcone may contain a hundred or more of them, so with many ranks this produces an enormous amount of redundant filesystem metadata traffic — each of the N files is opened N_ranks times.

For lightcones you can instead distribute whole files across ranks by passing :code:`mpi_mode="redshift"` to :py:meth:`open <opencosmo.open>`:

.. code-block:: python

   import opencosmo as oc

   lc = oc.open("step_600.hdf5", "step_601.hdf5", "step_602.hdf5", mpi_mode="redshift")

In this mode :code:`opencosmo` splits the redshift-ordered steps into contiguous chunks of roughly-equal data volume and assigns one chunk to each rank, so each rank owns a continuous redshift range and each file is opened by exactly one rank. (Low-redshift steps typically contain far fewer objects than high-redshift ones, so the chunk boundaries are chosen by row count rather than by an equal number of steps.) The rest of the API is unchanged: all operations still act on the local rank's data, scalar reductions are still combined globally, and :py:meth:`oc.write <opencosmo.write>` still coordinates across ranks to produce a single combined lightcone.

This also works for lightcone **structure collections**, where each redshift step is described by several linked files — for example halo properties alongside their particles and profiles. In that case a *whole step* (all of its linked files) is assigned to a single rank, so a halo and its particles and profiles always land together, and every rank builds a matching :py:class:`~opencosmo.StructureCollection`:

.. code-block:: python

   import opencosmo as oc

   sc = oc.open(
       "step_600/haloproperties.hdf5", "step_600/haloparticles.hdf5", "step_600/haloprofiles.hdf5",
       "step_601/haloproperties.hdf5", "step_601/haloparticles.hdf5", "step_601/haloprofiles.hdf5",
       mpi_mode="redshift",
   )

If there are fewer steps than ranks, the surplus ranks hold a full-schema, zero-length dataset (or structure collection) — every column, unit, and redshift range is still available on those ranks, so :code:`select`, :code:`filter`, and scalar reductions behave identically everywhere. Nested (Diffsky step-then-type) lightcones are not redshift-split in the current release; passing :code:`mpi_mode="redshift"` for them silently falls back to spatial distribution. Without an MPI communicator the argument is a no-op.

Combining Results Across Processes with :code:`reduce` and `gather`:
--------------------------------------------------------------------

Performing Multi-Rank Computations with :code:`reduce`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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



Combining Data with Gather
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you just want the column values, and don't need to perform an additional computation, you can use :py:meth:`gather <opencosmo.analysis.gather>`. :code:`gather` can retrieve columns that exist, or derived columns produced with :ref:`column expressions <Combining Columns Into New Columns>`. Like :code:`gather`, :code:`reduce` accepts a plotting function that can produce a plot of your result or perform another computation:

.. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        import opencosmo as oc
        from opencosmo.analysis import gather

        ds = oc.open("haloproperties.hdf5")

        def gas_mass_vs_halo_mass(sod_halo_mass, gas_frac, path)
                plt.scatter(sod_halo_mass, sod_halo_MGas)
                plt.semilogx()
                plt.xlabel("SOD Halo Mass")
                plt.ylabel("Halo Gas fraction")
                plt.savefig(path)
                

        gather(ds, 
                "sod_halo_gas", 
                gas_frac = oc.col("sod_halo_MGas") / oc.col("sod_halo_mass"), 
                format="numpy", 
                plotting_function = gas_mass_vs_halo_mas, 
                plotting_kwargs = {"path": "plots/gas_frac.png"
        )


Data is automatically sent to the root process, which performs the plotting. This function will work in a single-process environment as well, ensuring you can write scripts that run anywhere.



Important Caveats
-----------------


"Take" and "Select" Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a dataset is opened in an MPI context, the data is chunked across all ranks. :py:meth:`opencosmo.Dataset.take` operations will always operate on the data that is local to the given rank. For example, taking 100 rows at random on all ranks will actually take 100*N_ranks rows, distributed evenly across the ranks. Taking 100 rows with :code:`at = "start"` will take the first 100 rows on each rank.

When calling :py:meth:`select <opencosmo.Dataset.select>` or :py:meth:`drop <opencosmo.Dataset.drop>`, it is important to be sure to always include the same columns on all ranks if you intend to write data. If you attempt to write data and some processes have different columns than others, the write will fail.

When retrieving scalar values via :py:meth:`select <opencosmo.Dataset.select>`, computing derived columns via :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>`, or filtering via :py:meth:`filter <opencosmo.Dataset.filter>` in an MPI context, scalar reductions are combined across all ranks by default. To restrict the reduction to a single rank's local chunk, pass :code:`mode="local"` — see below.

Spatial Queries
~~~~~~~~~~~~~~~
In OpenCosmo, raw data is ordered according to its location in the spatial index. When a dataset is loaded with MPI, each rank recieves an equal share of the regions in the spatial index. As a result, most spatial queries are likely to return no data for most ranks. Ranks that fall completely outside the query region will return a zero-length dataset. :meth:`opencosmo.write` will handle the zero-length datasets automatically.

You can retrieve the region the local dataset is contained with in by calling :meth:`dataset.region <opencosmo.Dataset.region>`. One possible workflow is to perform different spatial queries for each rank depending on the region that is local to that rank.

Currently OpenCosmo does not support sharing data across ranks, such as when a given spatial query crosses a rank boundary. This will be improved in the future.

Global Scalar Reductions
-------------------------

When using :py:meth:`select <opencosmo.Dataset.select>` to compute scalar summary statistics (e.g. ``.min()``, ``.mean()``), the default behavior is :code:`mode="global"`, meaning the reduction is combined across all ranks before being returned. The same applies to scalar reductions used inside derived column expressions in :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>` or inside filter expressions in :py:meth:`filter <opencosmo.Dataset.filter>`. This way, every rank ends up with the same scalar value, which is usually what you want.

To restrict the reduction to each rank's own chunk of data — for per-rank statistics or when you intend to aggregate the results yourself — pass :code:`mode="local"`:

.. code-block:: python

   import opencosmo as oc

   ds = oc.open("haloproperties.hdf5")

   # All ranks receive the global minimum across the full dataset (default)
   global_min = ds.select(min_mass=oc.col("fof_halo_mass").min()).get_data()

   # Each rank returns its local minimum independently
   local_min = ds.select(
       min_mass=oc.col("fof_halo_mass").min(),
       mode="local",
   ).get_data()

Neither value of :code:`mode` has any effect when not running under MPI, or when the selection contains no scalar reductions.

Note that for the operations ``std``, ``var``, ``median``, and ``quantile``, the global reduction works by gathering all per-rank data to rank 0, computing the reduction there, and broadcasting the result back. For very large datasets this may be memory-intensive on rank 0.

The same :code:`mode` keyword is accepted by :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>` and :py:meth:`filter <opencosmo.Dataset.filter>` for scalars nested inside column arithmetic:

.. code-block:: python

   m = oc.col("fof_halo_mass")

   # Normalize each rank's data using the global mean and std (default)
   ds = ds.with_new_columns(zscore=(m - m.mean()) / m.std())

   # Filter against a globally-computed threshold so every rank uses the
   # same mean (default)
   ds = ds.filter(m > m.mean())

   # Same operations using each rank's own scalar
   ds = ds.with_new_columns(zscore=(m - m.mean()) / m.std(), mode="local")
   ds = ds.filter(m > m.mean(), mode="local")


