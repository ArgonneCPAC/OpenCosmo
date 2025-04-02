## OpenCosmo

The OpenCosmo Python Toolkit provides utilities for reading, writing and manipulating data from cosmological simulations produced by the Cosmolgical Physics and Advanced Computing (CPAC) group at Argonne National Laboratory. It can be used to work with smaller quantities data retrieved with the CosmoExplorer, as well as the much larget datasets these queries draw from. The OpenCosmo toolkit integrates with standard tools such as AstroPy, and allows uers to manipulate data in a fully-consistent cosmological context. k

### Installation

The OpenCosmo library is available for Python 3.11 and later and can be installed with `pip`:

```bash
pip install opencosmo
```

There's a good chance the default version of Python on your system is less than 3.11. Whether or not this is the case, we recommend installing `opencosmo` into a virtual environment. If you're using [https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html](Conda), you can create a new environment with Python 3.11 and install `opencosmo` into it like so:

```bash
conda create -n opencosmo python=3.11 opencosmo
conda activate opencosmo
```

This will create a new environment called `opencosmo` with Python 3.11 and install the `opencosmo` package into it with all necessary dependencies. If you plan to use opencosmo in a Jupyter notebook, you can install the `ipykernel` package to make the environment available as a kernel:

```bash
conda install ipykernel
python -m ipykernel install --user --name=opencosmo
```

Be sure you have run the "activate" command shown above before running the `ipykernel` command.

## Testing

To run tests, first download the test data [from Google Drive](https://drive.google.com/drive/folders/1CYmZ4sE-RdhRdLhGuYR3rFfgyA3M1mU-?usp=sharing). Set environment variable `OPENCOSMO_DATA_PATH` to the path where the data is stored. Then run the tests with `pytest`:

```bash
export OPENCOSMO_DATA_PATH=/path/to/data
# From the repository root
pytest
```
