Reading Data
============

OpenCosmo provides two functions for reading data from a file. :py:func:`opencosmo.read` and :py:func:`opencosmo.open`. Both return :py:class:`opencosmo.Dataset` objects. You should use :py:func:`opencosmo.read` if you're working with a small amount of data, and it's not a problem to load it all into memory at once. Use :py:func:`opencosmo.open` if the dataset is large and you'd rather not load it all into memory at once.

You can also use open as a context manager to automatically close the file when you're done with it:

.. code-block:: python

   import opencosmo as oc

   with oc.open("galaxyproperties.hdf5") as ds:
       print(ds.data)

Note About File Handles
-----------------------

OpenCosmo tries to avoid creating file handles whenever possible. When you transform a dataset with :py:meth:`opencosmo.Dataset.filter` or :py:meth:`opencosmo.Dataset.select`, the resulting dataset will share the same file handle as the original dataset. This means that if you call :py:meth:`opencosmo.Dataset.close` on the original dataset or any of its derivatives, the file will be closed for all of them and you will no longer be able to access the data.

If you do not close the dataset manually (or with a context manager) the underlying file handler will be closed when the last dataset using it goes out of scope.


.. autofunction:: opencosmo.read
.. autofunction:: opencosmo.open
.. autofunction:: opencosmo.read_cosmology
.. autofunction:: opencosmo.read_header
.. autofunction:: opencosmo.read_simulation_parameters
