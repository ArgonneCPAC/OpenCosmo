Reading and Writing Data in the OpenCosmo Format
================================================
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

Other Formats
-------------

OpenCosmo support dumping some data into other formats. These new files will not be readable by the toolkit, but may be more convinient for your specific usecase. You can install all additional io dependencies with

.. code-block:: bash

   pip install "opencosmo[io]"

or you can install individual output formats (see below).

Parquet
^^^^^^^

You can dump a :py:class:`Datast <opencosmo.Dataset>`, :py:class:`Lightcone <opencosmo.Lightcone>`, or parts of a :py:class:`StructureCollection <opencosmo.StructureCollection>` to parquet with :py:meth:`opencosmo.io.write_parquet`. You will need to install pyarrow with parquet support first:

.. code-block:: bash

        pip install "pyarrow[parquet]"


A dataset will simply be dumped as a collection of columns. Any querying (selection, filtering, etc.) will persist into the output. Metadata such as unit information and the spatial index will not be included:

.. code-block:: python

        import opencosmo as oc
        from opencosmo.io import write_parquet

        dataset = oc.open("haloproperties.hdf5")
        write_parquet("my_dataset.parquet", dataset)


You can also write the particles of a :py:class:`StructureCollection <opencosmo.StructureCollection>`. 

.. code-block:: python

   structures = oc.open("haloproperties.hdf5", "haloparticles.hdf5")
   write_parquet("my_structure/", structures)

This will produce one parquet file for each particle type in the collection. 


