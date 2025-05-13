ARG MPI_IMPL="mpich"

FROM astropatty/phdf5:latest-${MPI_IMPL}

RUN python -m pip install opencosmo

