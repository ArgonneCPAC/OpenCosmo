FROM python:3.13

ARG HDF5_VERSION="1.14.6"
ARG HDF5_LIB="https://github.com/HDFGroup/hdf5/releases/download/hdf5_${HDF5_VERSION}/hdf5.tar.gz"
ARG MPI_IMPL="mpich"

RUN apt-get update && apt-get -y install build-essential ${MPI_IMPL} wget make

WORKDIR /install
RUN wget ${HDF5_LIB} && tar -xvzf hdf5.tar.gz && mv hdf5-${HDF5_VERSION} hdf5
WORKDIR /install/hdf5
RUN CC=/usr/bin/mpicc ./configure --prefix /usr/local --enable-parallel && make -j4 && make install

WORKDIR /app
RUN rm -rf /install
RUN CC=mpicc HDF5_MPI="ON" HDF5_DIR="/usr/local" python -m pip install --no-binary=h5py h5py

RUN python -m pip install opencosmo

