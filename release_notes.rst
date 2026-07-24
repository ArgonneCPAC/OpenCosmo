opencosmo 1.3.11 (2026-07-23)
=============================

Bugfixes
--------

- Fixed a failure when writing a lightcone after dropping its angular coordinate columns (`ra`/`dec` or `theta`/`phi`). (268a)
- Fixed a `KeyError` when writing a dataset after dropping a derived column, such as `top_host_idx` in diffsky data. (268b)



