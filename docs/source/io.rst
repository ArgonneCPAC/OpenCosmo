Reading and Writing Data
========================
OpenCosmo defines a data format for storing simulation data in hdf5 files. A dataset in this format can be transformed and written by the OpenCosmo library to produce a new file that can be read by others (or yourself at a later date!) using the library.

Options for Reading Data
------------------------

Any single opencosmo file can be open with :py:meth:`opencosmo.open`. This function will parse the structure of the file and return the appropriate object, such as a :py:class:`opencosmo.Dataset` or :py:class:`opencosmo.StructureCollection`

.. code-block:: python

   import opencosmo as oc

   ds = oc.open("haloproperties.hdf5")

You can also use open as a context manager to automatically close the file when you're done with it:

.. code-block:: python

   import opencosmo as oc
   with oc.open("galaxyproperties.hdf5") as ds:
       print(ds.data)


When opening multiple files that are linked to each other, use :py:meth:`opencosmo.open`

Writing Data
------------

Writing data to a new file is straightforward:

.. code-block:: python

   oc.write("my_output.hdf5", ds)

Transformations applied to the data will propogate to the file when written, with the exception of :py:meth:`oc.Dataset.with_units`. When you write data, it will always be stored in the unit convention of the original raw data.


