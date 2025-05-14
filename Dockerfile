ARG MPI_IMPL="mpich"

FROM astropatty/parallel-h5py:3.13.0-${MPI_IMPL}

RUN python -m pip install opencosmo

