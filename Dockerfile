ARG MPI_IMPL="mpich"

FROM astropatty/mpipy:mpich
COPY . /app/deps/opencosmo

RUN pip install -e /app/deps/opencosmo
RUN pip install mpi4py

