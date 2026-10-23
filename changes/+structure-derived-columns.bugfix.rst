Derived columns added to linked datasets in a ``StructureCollection`` are now still
cached during iteration after ``with_units`` or ``with_new_columns`` on a nested
collection. Previously these calls caused such columns to be recomputed after
iterating with ``objects()``.
