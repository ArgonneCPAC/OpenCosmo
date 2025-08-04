# Contributing to OpenCosmo

We welcome contributions to the OpenCosmo toolkit from the greater astrophysics community. We especially appreciate contributions that arise from 

## Feature Requests/Bug Reports

Feature requests and bug reports should be submitted as issues on the main OpenCosmo repo. Before raising an issue, be sure to check through existing issues to ensure it hasn't already been suggested by someone else. Feel free to leave comments on existing issues to add additional context, suggest adjustments, or just let us know you care about it. We will seek to prioritize features that will be useful to as many people as possible. 

## Contributing Documentation

If you find that the documentation is not quite up to par, feel free to create a documentation-only pull request. Most of the steps below do not apply in this case, except for Step 6.

## Contributing Code

In addition to raising issues on the repository or adding documentation, we welcome PRs that fix bugs, improve performance, or add new features. The remainder of this document describes how to go about setting up a development environment and prepping your PR for merging.

### Step 0: Install UV

We use [uv](https://docs.astral.sh/uv/) to manage dependencies and provide a consistent execution environment for packaging and testing. If you do not already have uv installed on your machine, [follow the documentation](https://docs.astral.sh/uv/getting-started/installation/) for your system.

### Step 1: Create a Fork of the OpenCosmo Repo and Clone

You should preform your development in your own personal fork of the repository. You can create one [at this link](https://github.com/ArgonneCPAC/OpenCosmo/fork). Once you have a fork, you can clone it to the machine you will be doing your development on.

### Step 2: Create a Virtual Environment with UV

In addition to managing dependencies, uv manages a virtual environment with the appropriate versions of all packages. From the root of the repository, run the command:

```bash
uv sync
```

This will create a virtual environment in the repository and install all necessary dependencies, as well as extra dependencies required for development work.

### Step 2.1: Install Parallel HDF5 (Optional)

If you plan to work on features that involve parallel I/O, you will need to install parallel HDF5 to run parallel tests. To start, install the additional packages necessary for developing and testing parallel features
```bash
uv sync --group mpi --group test-mpi

```
Next, you will need to install parallel HDF5 itself. The exact steps will depend on your machine and package manager, but should end with a command that looks like the following:

```bash
CC="mpicc" HDF5_MPI="ON" uv pip install --reinstall --no-binary=h5py h5py
```

Note that running with uv is required to ensure that the parallel version of HDF5 is properly installed into the virtual environment and is used by UV when running tests. This is required even if you have activated the virtual envionment. This command requires that you have a copy of parallel HDF5 available on your machine. 

### Step 3: Add a Commit & Create a PR

I know what you're thinking: I haven't actually implemented my changes yet. Why am I already submitting a PR? Two reasons. First, so we know what people are working on so we're not doing duplicate work. Second, so that your PR starts running through the CI pipeline.

### Step 4: Implement Your Changes

This is where the magic happens. Implement your features! If you have questions, feel free to tag @AstroPatty

### Step 5: Lint and Write Tests

Our CI pipeline performs linting with [ruff](https://astral.sh/ruff) and static type checking with [mypy](https://www.mypy-lang.org/). Both should have been installed automatically when you ran `uv sync`. Your PR must pass both to be merged. If you prefer, you can use [pre-commit](https://pre-commit.com/) to automatically run linting and type checking before you commit. Altenatively, you can call them manually on your last commit. 

Many libraries do not have full typing support. If this is the case, you can add a `# type: ignore` directive when you import them. If the type stubs exist as a seperate library, you should instead add them as a development dependency in the pyprojet.toml with 

```bash

uv add --dev (package_type_library_name)
```

If your PR is implenting a new feature, you should also add tests that ensure your feature continues to work as expected. These tests will be run on a small group of test data, which can be downloaded from the [OpenCosmo Google Drive](https://drive.google.com/drive/folders/1CYmZ4sE-RdhRdLhGuYR3rFfgyA3M1mU-?usp=sharing). Check the existing tests for deatils about how to request paths to the test data in your tests. 

If your feature can raise errors, those errors should include easy-to-understand messages that explain what went wrong. This is most common when the user provides inputs that are invalid, such as requesting a column that does not exist in the dataset. It may be worthwhile to write a test that ensures this error is raised with the invliad input. For more information, see the [pytest documentation](https://docs.pytest.org/en/stable/how-to/assert.html#assertraises)

Once your update has been made, you can run the tests with:

```bash
uv run pytest --ignore=test/parallel
```

If you are working on features designed to be used in an MPI context, you can run the parallel tests with:

```bash
uv run mpiexec -n 4 pytest -m parallel test/parallel -x
```

All tests must pass for your PR to be merged.

### Step 6: Add Documentation and a Changelog

If your changes need user documentation (i.e. are not just a bugfix or behind the-scenes improvement) you should add documentation as appropriate. We use [sphinx](sphinx-doc.org) to manage documentation.

Whether or not you are adding documentation, you should add one or multiple news blurbs to your PR in /changes that describe the changes that were made. We use [towncrier](https://towncrier.readthedocs.io/en/latest/tutorial.html) to manage news blurbs, which will be automatically added to a unified changelog when a new version is released. If your change is related to a specific issue, the issue number should be included in the file name. We recommend using the built in `towncrier create` CLI to create your news blurbs

### Step 5: Request a Review

Once you are happy with your PR, request a review. We will be by to check it and (potentially) provide feedback. Note that you should feel free to reach out to us before you get to this stage! It's almost always easier to get things merged at the end if there's been an ongoing conversation beforehand.

### Step 6: Sit Back and Enjoy a Job Well Done  

Once your PR is merged, it be made available publicly on the next numbered release. Sit back and enjoy your handywork!
