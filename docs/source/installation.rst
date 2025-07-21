Installation
============

The OpenCosmo library is available for Python 3.11 and later on Linux and MacOS (and Windows via `WSL <https://learn.microsoft.com/en-us/windows/wsl/setup/environment>`_). It can be installed with :code:`pip`:

.. code-block:: bash

   pip install opencosmo

There's a good chance the default version of Python on your system is less than 3.11. Whether or not this is the case, we recommend installing :code:`opencosmo` into a virtual environment. If you're using `Conda <https://docs.conda.io/projects/conda/en/stable/:code:user-guide/getting-started.html>`_ you can create a new environment and install :code:`opencosmo` into it automatically:

.. code-block:: bash

   conda create -n opencosmo_env python=3.11 conda-forge::opencosmo


or, if you already have a virtual environment you'd like to use:

.. code-block:: bash 

   conda install conda-forge::opencosmo

If you plan to use :code:`opencosmo` in a Jupyter notebook, you can install the :code:`ipykernel` package to make the environment available as a kernel:

.. code-block:: bash

   conda install ipykernel
   python -m ipykernel install --user --name=opencosmo

Be sure you have run the :code:`activate` command shown above before running the :code:`ipykernel` command.

If you are interested in the tools in the :doc:`analysis <analysis>` module, they need to be installed seperately:

.. code-block:: bash

   opencosmo install haloviz


Note that the analysis tools must be installed with :code:`pip`

Installing with MPI Support
---------------------------

:code:`opencosmo` can leverage MPI to distribute analysis on a very large dataset across multiple cores or nodes. You simply need to install the :code:`mpi4py` package:

.. code-block:: bash

   pip install mpi4py

:code:`opencosmo` Can both read and write data in an MPI context with no additional setup. By default, ranks must write to the file one at a time. This may result in poor performance if the data being written is large. This can be improved by installing the :code:`h5py` package with MPI support. Pre-built wheels with MPI support are not generally available on PyPI, so you will need to build :code:`h5py` from source against a version of the HDF5 library that was built with MPI support. Many HPC systems have an optimized MPI-enabled version of HDF5 available. For example, on Polaris at the Argonne Leadership Computing Facility (ALCF), run the follwing commands in the Python environment you plan to use with :code:`opencosmo`:

.. code-block:: bash

   pip uninstall h5py # If a non-MPI version is already installed in your environment
   HDF5_MPI="ON" HDF5_DIR=$HDF5_DIR  CC="cc" pip install --no-cache-dir --force-reinstall --no-binary=h5py h5py

Parallel HDF5 is also available at `NERSC <https://docs.nersc.gov/development/languages/python/parallel-python/#parallel-io-with-h5py>`_ and `OLCF <https://docs.olcf.ornl.gov/software/python/parallel_h5py.html>`_. See the linked documentation for details of getting your environment set up at one of those facilities.

