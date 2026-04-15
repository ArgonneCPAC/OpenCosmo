opencosmo 1.2.5 (2026-04-15)
============================

Bugfixes
--------

- Fix a bug that could cause column expressions to fail when they contain an astropy unit


New Features
------------

- :py:func:`reduce <opencosmo.analysis.reduce>` now accepts a `plotting_function` argument, which allows production of a plot after reduction is performed.


Improvements
------------

- Data handlers and cache are now protocol-based to allow for new backends



