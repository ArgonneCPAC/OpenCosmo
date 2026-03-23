opencosmo 1.2.3 (2026-03-23)
============================

Deprecations and Removals
-------------------------

- :py:meth:`evaluate <opencosmo.Dataset.evaluate>` no longer broadcasts array keyword arguments that have the same length as the dataset. You can still use arrays in this way by setting :code:`vectorize = True` or first inserting the array as a new column with :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>`



