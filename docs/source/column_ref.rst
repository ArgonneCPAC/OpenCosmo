
Columns
=======

In OpenCosmo, references to columns are created independently of the datasets that contain them. You can
create combinations of columns with basic arithmetic, as well as ...

Columns created this way can be added to datasets or collections with the :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>` method. The actual *values* in these columns are evaluated lazily, so it is fine to create these new columns at the beginning of your analysis even if you plan to filter out significant numbers of the rows.

.. autofunction:: opencosmo.col

.. autoclass:: opencosmo.column.Column
   :members:

Provided Column Combinations
----------------------------

There are a number of basic column combinations that are 

.. autofunction:: opencosmo.column.norm_cols
.. autofunction:: opencosmo.column.add_mag_cols
.. autofunction:: opencosmo.column.offset_3d
