Reading and Writing Data
========================
OpenCosmo defines a data format for storing simulation data in hdf5 files. A dataset in this format can be transformed and written by the OpenCosmo library to produce a new file that can be read by others (or yourself at a later date!) using the library.

Options for Reading Data
------------------------

OpenCosmo provides two functions for reading data from a file. :py:func:`opencosmo.read` and :py:func:`opencosmo.open`. Both return :py:class:`opencosmo.Dataset` objects. The only difference is that :py:func:`opencosmo.read` immediately loads all data into memory, while :py:func:`opencosmo.open` keeps an open file handle and only loads the data from the file when it is actually neeeded.

You can also use open as a context manager to automatically close the file when you're done with it:

.. code-block:: python

   import opencosmo as oc

   with oc.open("galaxyproperties.hdf5") as ds:
       print(ds.data)

Some collections can only be opened with :py:func:`opencosmo.open`, so we recommend it for general usage.

Writing Data
------------

Writing data to a new file is straightforward:

.. code-block:: python

   oc.write("my_output.hdf5", ds)

Transformations applied to the data will propogate to the file when written, with the exception of :py:meth:`oc.Dataset.with_units`. Data is always stored in the scale-free unit convention, and can be readily transformed to a new convention once it is read.


