Added ``opencosmo.serde`` messages for ``StructureCollection`` transformations
(``filter``, ``select``, ``drop``, ``sort_by``, ``take``, ``take_range``, ``take_rows``,
``bound``, ``with_new_columns``, ``with_units``, ``with_datasets``,
``with_redshift_range`` and ``pixel_search``), generated from the method signatures.
``select``, ``drop`` and ``with_units`` accept nested per-dataset specifications.
``opencosmo.serde.execute_message`` now accepts a ``StructureCollection`` target, and
variadic keyword names that collide with explicit method parameters are rejected
when a message is parsed.
