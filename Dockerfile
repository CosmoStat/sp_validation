# Development image with more bells and whistles
FROM ghcr.io/cosmostat/shapepipe:develop

RUN apt-get update -y --quiet --fix-missing && \
    apt-get dist-upgrade -y --quiet --fix-missing && \
    apt-get install -y --quiet \
        libgsl-dev \
        htop \
        npm \
        tmux

RUN pip install --no-cache-dir \ 
    snakemake

WORKDIR /sp_validation
COPY . /sp_validation

# Install sp_validation
RUN pip install --no-cache-dir -e .
