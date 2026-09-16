Matching Halos Across Simulations
=================================

A :py:class:`SimulationCollection <opencosmo.SimulationCollection>` normally holds
independent catalogs from several simulations. Rows in those catalogs do not have to
refer to the same halo, even when they are in the same position. Dataset matching
uses a supplied mapping file to align corresponding halos, making direct
simulation-to-simulation comparisons possible.

Matching is available for datasets with a precomputed one-to-one halo mapping. OpenCosmo
cannot perform matching if this information is not available.

Opening Matched Catalogs
------------------------

Pass the halo-property catalogs and their mapping file to :py:func:`opencosmo.open`.
The result is a :py:class:`SimulationCollection <opencosmo.SimulationCollection>`.
The catalog names come from the simulation metadata and are available through
``collection.keys()``. For readability, the examples below refer to the three
SCIDAC catalogs as ``SCIDAC_001``, ``SCIDAC_002``, and ``SCIDAC_003`` rather
than their full metadata names.

.. code-block:: python

   import opencosmo as oc

   catalogs = oc.open(
       "haloproperties_go.hdf5",
       "scidac_000/haloproperties.hdf5",
       "scidac_001/haloproperties.hdf5",
       "scidac_002/haloproperties.hdf5",
       "halo_mapping.hdf5",
   )

   print({name: len(dataset) for name, dataset in catalogs.items()})

.. code-block:: text

   {'SCIDAC_001': 237116,
    'SCIDAC_002': 236807,
    'SCIDAC_003': 236530,
    'SCIDAC_128_GO': 226670}

The mapping file can include mappings for more simulations than were opened. OpenCosmo
uses only mappings whose endpoints are among the supplied catalogs. A mapping file
cannot be opened by itself.

If you later write this collection out, all data (including the matching) will be written
into a single file.

Aligning Halos
--------------

Call :py:meth:`match <opencosmo.SimulationCollection.match>` with the name of the
catalog that should drive the order. The resulting collection contains only halos
with a corresponding halo in *every* opened catalog. All datasets have equal length,
and row ``i`` in each dataset describes the same matched halo.

.. code-block:: python

   matched = catalogs.match("SCIDAC_128_GO")
   print({name: len(dataset) for name, dataset in matched.items()})

.. code-block:: text

   {'SCIDAC_128_GO': 197006,
    'SCIDAC_001': 197006,
    'SCIDAC_002': 197006,
    'SCIDAC_003': 197006}

Halo tags are catalog-local identifiers, so they generally differ between matched
rows. The following preview, produced from the test catalogs above, shows that the
first five rows are aligned even though each catalog has different tags and slightly
different halo masses:

.. code-block:: python

   preview = matched.select("fof_halo_tag", "fof_halo_mass").take(5)
   for name, dataset in preview.items():
       print(name)
       print(dataset.get_data(format="numpy"))

.. code-block:: text

   SCIDAC_128_GO
   {'fof_halo_tag': array([27133948, 95228940, 31616069, 14931717, 94357410]),
    'fof_halo_mass': array([1.1113736e+11, 2.4807445e+11, 2.3735763e+12,
                             8.3353018e+10, 8.7322206e+10], dtype=float32)}

   SCIDAC_001
   {'fof_halo_tag': array([ 55315440, 186787862,  63751292,  30388748, 191331144]),
    'fof_halo_mass': array([1.1026644e+11, 2.0382586e+11, 2.0549656e+12,
                             7.8523073e+10, 8.1864483e+10], dtype=float32)}

   SCIDAC_002
   {'fof_halo_tag': array([ 54263788, 185212944,  61130888,  29341202, 189239106]),
    'fof_halo_mass': array([1.1026644e+11, 2.0549655e+11, 2.0649898e+12,
                             7.8523073e+10, 7.6852371e+10], dtype=float32)}

   SCIDAC_003
   {'fof_halo_tag': array([ 54267892, 188886034,  61127822,  28291598, 189239106]),
    'fof_halo_mass': array([1.0191293e+11, 2.0382586e+11, 2.0449414e+12,
                             7.8523073e+10, 7.6852371e+10], dtype=float32)}

Source-Driven Transformations
-----------------------------

After calling ``match()``, row transformations operate on the catalog passed to
``match()``. OpenCosmo applies the corresponding row selection and ordering to every
other catalog. This lets you define a sample in one simulation while retaining its
matched halos elsewhere.

.. code-block:: python

   massive = (
       catalogs
       .match("SCIDAC_128_GO")
       .filter(oc.col("fof_halo_mass") > 1e15)
       .select("fof_halo_tag", "fof_halo_mass")
   )

   print({name: len(dataset) for name, dataset in massive.items()})

.. code-block:: text

   {'SCIDAC_128_GO': 2,
    'SCIDAC_001': 2,
    'SCIDAC_002': 2,
    'SCIDAC_003': 2}

In this example, the mass threshold is evaluated only against ``SCIDAC_128_GO``.
The corresponding halos are retained in the other simulations even when their own
``fof_halo_mass`` values are below the threshold. The same source-driven behavior
applies to :py:meth:`filter <opencosmo.SimulationCollection.filter>`,
:py:meth:`bound <opencosmo.SimulationCollection.bound>`,
:py:meth:`sort_by <opencosmo.SimulationCollection.sort_by>`,
:py:meth:`take <opencosmo.SimulationCollection.take>`, and
:py:meth:`take_range <opencosmo.SimulationCollection.take_range>`.

The source determines the final row order. For example, sorting a matched collection
by mass sorts the source catalog and reorders every target catalog to preserve
row-wise correspondence. Call ``match()`` before ``sort_by()``; matching catalogs
with an existing sort order is not supported.

Selections, unit-convention changes, and derived columns do not change the active
source. You can use them before or after matching. Filters and other row selections
applied before ``match()`` are also respected: a correspondence is retained only when
both its source and target rows remain in their respective catalogs.

Matching with MPI
-----------------

Under MPI, OpenCosmo initially partitions each catalog spatially across ranks. A
matched target halo does not necessarily belong to the same rank as its source halo.

When :py:meth:`match <opencosmo.SimulationCollection.match>` is called, OpenCosmo
keeps each rank's selected source rows and exchanges the corresponding rows in the
target datasets between ranks.

Matching is therefore a collective operation and can communicate data even though
ordinary transformations usually operate only on local rank data. Every rank that
opened the collection must call ``match()`` in the same order. The final per-rank row
counts may differ from the original spatial partitioning, because they follow the
distribution of the active source catalog rather than the target catalogs.

Returning to Independent Catalogs
---------------------------------

Use :py:meth:`clear_match <opencosmo.SimulationCollection.clear_match>` when later
operations should again apply independently to every catalog:

.. code-block:: python

   matched = catalogs.match("SCIDAC_128_GO")
   matched_massive = matched.filter(oc.col("fof_halo_mass") > 1e15)

   independent = matched_massive.clear_match()
   # Subsequent filters, selections, and row operations apply to every catalog.

``clear_match()`` does not restore rows removed by matching or by earlier
transformations. It only removes the source-driven behavior for later operations.
