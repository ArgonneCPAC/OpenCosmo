<h1 align="center">
    <picture>
        <source srcset="https://raw.githubusercontent.com/ArgonneCPAC/opencosmo/main/branding/opencosmo_dark.png" media="(prefers-color-scheme: dark)">
        <source srcset="https://raw.githubusercontent.com/ArgonneCPAC/opencosmo/main/branding/opencosmo_light.png" media="(prefers-color-scheme: light)">
        <img src="https://raw.githubusercontent.com/ArgonneCPAC/opencosmo/main/branding/opencosmo_light.png" alt="OpenCosmo">
    </picture>
</h1><br>

[![CI](https://github.com/ArgonneCPAC/OpenCosmo/actions/workflows/merge.yaml/badge.svg)](https://github.com/ArgonneCPAC/OpenCosmo/actions/workflows/merge.yaml)
![PyPI - Version](https://img.shields.io/pypi/v/opencosmo)
![Conda Version](https://img.shields.io/conda/vn/conda-forge/opencosmo)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/opencosmo)
![GitHub License](https://img.shields.io/github/license/ArgonneCPAC/opencosmo)


The OpenCosmo Python Toolkit provides utilities for reading, writing and manipulating data from cosmological simulations produced by the Cosmolgical Physics and Advanced Computing (CPAC) group at Argonne National Laboratory. It can be used to work with smaller quantities data retrieved with the CosmoExplorer, as well as the much larget datasets these queries draw from. The OpenCosmo toolkit integrates with standard tools such as AstroPy, and allows you to manipulate data in a fully-consistent cosmological context.

### Installation

The OpenCosmo library is available for Python 3.11 and up on Linux and MacOS (and Windows via [WSL](https://learn.microsoft.com/en-us/windows/wsl/setup/environment)). It can be installed easily with `pip`:

```bash
pip install opencosmo
```


There's a good chance the default version of Python on your system is less than 3.11. Whether or not this is the case, we recommend installing `opencosmo` into a virtual environment. If you're using [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html), you can create a new environment and install `opencosmo` into it automatically:

```bash
conda create -n opencosmo_env conda-forge::opencosmo
conda activate opencosmo_env
```

or if you already have a virtual environment to use:

```bash
conda install conda-forge::opencosmo
```

If you plan to use `opencosmo` in a Jupyter notebook, you can install the `ipykernel` package to make the environment available as a kernel:

```bash
pip install ipykernel # can also be installed with conda
python -m ipykernel install --user --name=opencosmo
```

Be sure you have run the "activate" command shown above before running the `ipykernel` command.

## Getting Started

To get started, download the "haloproperites.hdf5" from the [OpenCosmo Google Drive](https://drive.google.com/drive/folders/1CYmZ4sE-RdhRdLhGuYR3rFfgyA3M1mU-?usp=sharing). This file contains properties of dark-matter halos from a small hydrodynamical simulation run with HACC. You can easily open the data with the `open` command:

```python
import opencosmo as oc

dataset = oc.open("haloproperties.hdf5")
print(dataset)
```

```text
OpenCosmo Dataset (length=237441)
Cosmology: FlatLambdaCDM(name=None, H0=<Quantity 67.66 km / (Mpc s)>, Om0=0.3096446816186967, Tcmb0=<Quantity 0. K>, Neff=3.04, m_nu=None, Ob0=0.04897468161869667)
First 10 rows:
block fof_halo_1D_vel_disp fof_halo_center_x ... sod_halo_sfr unique_tag
             km / s               Mpc        ... solMass / yr
int32       float32             float32      ...   float32      int64
----- -------------------- ----------------- ... ------------ ----------
    0            32.088795         1.4680439 ...       -101.0      21674
    0             41.14525        0.19616994 ...       -101.0      44144
    0             73.82962         1.5071135 ...    3.1447952      48226
    0             31.17231         0.7526525 ...       -101.0      58472
    0            23.038841         5.3246417 ...       -101.0      60550
    0            37.071426         0.5153746 ...       -101.0     537760
    0            26.203058         2.1734374 ...       -101.0     542858
    0              78.7636         2.1477687 ...          0.0     548994
    0             37.12636         6.9660196 ...       -101.0     571540
    0             58.09235          6.072006 ...    1.5439711     576648
```

The `open` function returns a `Dataset` object, which holds the raw data as well as information about the simulation. You can easily access the data and cosmology as Astropy objects:
```python
dataset.data
dataset.cosmology
```

The first will return an astropy table of the data, with all associated units already applied. The second will return the astropy cosmology object that represents the cosmology the simulation was run with. 

### Basic Querying

Although you can access data directly, `opencosmo` provides tools for querying and transforming the data in a fully cosmology-aware context. For example, suppose we wanted to plot the concentration-mass relationship for the halos in our simulation above a certain mass. One way to perform this would be as follows:

```python
dataset = dataset
    .filter(oc.col("fof_halo_mass") > 1e13)
    .take(1000, at="random")
    .select(("fof_halo_mass", "sod_halo_cdelta"))

print(dataset)

```

```text
OpenCosmo Dataset (length=1000)
Cosmology: FlatLambdaCDM(name=None, H0=<Quantity 67.66 km / (Mpc s)>, Om0=0.3096446816186967, Tcmb0=<Quantity 0. K>, Neff=3.04, m_nu=None, Ob0=0.04897468161869667)
First 10 rows:
 fof_halo_mass   sod_halo_cdelta
    solMass
    float32          float32
---------------- ---------------
11220446000000.0       4.5797048
17266723000000.0       7.4097505
51242150000000.0       1.8738283
70097712000000.0       4.2764015
51028305000000.0        2.678151
11960567000000.0       3.9594727
15276915000000.0        5.793542
16002001000000.0       2.4318497
47030307000000.0       3.7146702
15839942000000.0        3.245569
```

We could then plot the data, or perform further transformations. This is cool on its own, but the real power of `opencosmo` comes from its ability to work with different data types. Go ahead and download the "haloparticles" file from the [OpenCosmo Google Drive](https://drive.google.com/drive/folders/1CYmZ4sE-RdhRdLhGuYR3rFfgyA3M1mU-?usp=sharing) and try the following:

```python
import opencosmo as oc

data = oc.open("haloproperties.hdf5", "haloparticles.hdf5")
```
This will return a data *collection* that will allow you to query and transform the data as before, but will associate the halos with their particles. 

```python

data = data
    .filter(oc.col("fof_halo_mass") > 1e13)
    .take(1000, at="random")

for halo in data.halos():
    halo_properties = halo["halo_properties"]
    dm_particles = halo["dm_particles"]
    star_particles = halo["star_particles"]
```

In each iteration, "halo properties" will be a dictionary containing the properties of the halo (such as its total mass), while "dm_particles" and "star_particles" will be OpenCosmo datasets containing the dark matter and stars associated with the halo, respectively. Because these are just like the dataset object we saw eariler, we can further query and transform the particles as needed for our analysis. For more details on how to use the library, check out the [full documentation](https://opencosmo.readthedocs.io/en/latest/).

### Testing

To run tests, first download the test data [from Google Drive](https://drive.google.com/drive/folders/1CYmZ4sE-RdhRdLhGuYR3rFfgyA3M1mU-?usp=sharing). Set environment variable `OPENCOSMO_DATA_PATH` to the path where the data is stored. Then run the tests with `pytest`:

```bash
export OPENCOSMO_DATA_PATH=/path/to/data
# From the repository root
pytest --ignore test/parallel 
```

Although opencosmo does support multi-core processing via MPI, the default installation does not include the necessary dependencies to work in an MPI environment. If you need these capabilities, check out the guide in our documentation.

### Contributing

We welcome bug reports and feature requests from the community. If you would like to contribute to the project, please check out the [contributing guide](CONTRIBUTING.md) for more information.

```
