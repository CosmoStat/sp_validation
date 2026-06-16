# Development image with more bells and whistles
FROM ghcr.io/cosmostat/shapepipe:im_sims

RUN apt-get update -y --quiet --fix-missing && \
    apt-get dist-upgrade -y --quiet --fix-missing && \
    apt-get install -y --quiet \
        autoconf \
        automake \
        libtool \
        pkg-config \
        htop \
        npm \
        tmux

# The base image runs out of a uv venv (VIRTUAL_ENV=/app/.venv, where shapepipe
# lives); bare `pip` falls through to /usr/local and installs into the wrong
# interpreter. Use `uv pip` so packages land in the venv that `python` resolves to.
RUN uv pip install --no-cache-dir \
    snakemake

WORKDIR /sp_validation
COPY . /sp_validation

# Install with the test extra so the image can run the unit suite in CI.
RUN uv pip install --no-cache-dir -e '.[test]'
