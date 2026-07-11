# Development image with more bells and whistles
FROM ghcr.io/cosmostat/shapepipe:develop

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

# The base shapepipe image provides a uv-managed venv at /app/.venv (exported as
# VIRTUAL_ENV); install sp_validation's deps into that same venv rather than
# spawning a second one under /sp_validation.
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /sp_validation

# uv.lock is the SSOT: `uv sync --frozen` installs exactly what it pins, so an
# image build can never silently re-resolve and drift a base-image version (the
# numpy-past-numba drift this lockfile exists to prevent). `--inexact` keeps the
# base image's ShapePipe stack (shapepipe, ngmix, galsim, …) — packages not in
# our lock — instead of pruning them. Copy the lock + manifest first so this
# layer caches independently of source edits. Extras: test (CI unit suite),
# glass (GLASS map-level mock — pulls glass.ext.camb + the cosmology wrapper),
# workflow (Snakemake + mpi4py runners). cs_util 0.2.2 (with cs_util.size) and a
# numba-safe numpy 2.4.6 come straight from the lock, so the old ad-hoc snakemake
# and cs_util `--upgrade` layers are gone.
COPY pyproject.toml uv.lock /sp_validation/
RUN uv sync --frozen --inexact --no-install-project \
    --extra test --extra glass --extra workflow

# Install sp_validation itself (editable) into the same venv; deps are already
# satisfied by the sync above.
COPY . /sp_validation
RUN uv pip install --no-deps -e .
