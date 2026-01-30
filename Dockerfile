# Development image with more bells and whistles
FROM ghcr.io/cosmostat/shapepipe:py312

RUN apt-get update -y --quiet --fix-missing && \
    apt-get dist-upgrade -y --quiet --fix-missing && \
    apt-get install -y --quiet \
        libcfitsio-dev \
        libfftw3-dev \
        libgsl-dev \
        libhealpix-cxx-dev \
        htop \
        npm \
        tmux

RUN pip install --no-cache-dir \ 
    snakemake

WORKDIR /sp_validation
COPY . /sp_validation

# Install sp_validation
# Set paths for pymaster to find cfitsio
ENV CFITSIO_DIR=/usr
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
RUN pip install --no-cache-dir -e .
