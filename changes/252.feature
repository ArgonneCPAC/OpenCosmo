:py:meth:`opencosmo.col` expressions now support scalar reduction methods: ``.mean()``, ``.min()``, ``.max()``, ``.std()``, ``.var()``, ``.median()``, ``.sum()``, and ``.quantile(q)``. Scalar reductions can be used in column arithmetic (e.g. normalization) and in filter expressions:

.. code-block:: python

   m = oc.col("fof_halo_mass")

   # Normalize a column
   ds = ds.select("*", scaled=(m - m.min()) / (m.max() - m.min()))

   # Filter relative to a data-driven threshold
   ds = ds.filter(m < m.mean())
