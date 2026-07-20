opencosmo 1.3.10 (2026-07-20)
=============================

Bugfixes
--------

- Fix a ``StopIteration`` when writing a sorted structure collection whose links are all of the start/size kind (e.g. halo properties + halo particles) with no idx-style link.
- Fix a ``TypeError: len() of unsized object`` when doing a global sorted ``take`` in an MPI environment where a rank holds a single-row dataset, which caused the sort column to collapse to a scalar.



