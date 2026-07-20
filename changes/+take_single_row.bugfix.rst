Fix a ``TypeError: len() of unsized object`` when doing a global sorted ``take`` in an MPI environment where a rank holds a single-row dataset, which caused the sort column to collapse to a scalar.
