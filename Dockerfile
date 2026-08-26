# Development image with more bells and whistles
FROM ghcr.io/cosmostat/shapepipe:im_sims

# liblapack-dev: cosmosis's MultiNest links -llapack, and the base image ships
# only the runtime liblapack.so.3 (no dev symlink).
RUN apt-get update -y --quiet --fix-missing && \
    apt-get dist-upgrade -y --quiet --fix-missing && \
    apt-get install -y --quiet \
        autoconf \
        automake \
        libtool \
        pkg-config \
        htop \
        npm \
        tmux \
        liblapack-dev

# The base shapepipe image provides a uv-managed venv at /app/.venv (exported as
# VIRTUAL_ENV); install sp_validation's deps into that same venv rather than
# spawning a second one under /sp_validation.
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
# $HOME is bind-mounted under apptainer, so uv would otherwise discover the
# host's managed CPythons -- including newer ones that satisfy requires-python
# -- and build a venv against an interpreter carrying none of this stack.
ENV UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never

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

# cosmosis builds MPI-enabled polychord/multinest only when MPIFC is set: its
# setup.py exports MPIFC for conda builds only, and the sampler Makefiles gate on
# `which $(MPIFC)`. Absolute path, not a bare name: /opt/ompi/bin is not always on
# PATH, and a miss silently omits libchord_mpi.so while the install still succeeds,
# and `cosmosis --mpi` fails at load time -- which is how the pipeline runs, since
# the --smp pool is broken upstream. Must precede the sync that builds cosmosis.
ENV MPIFC=/opt/ompi/bin/mpif90

RUN uv sync --frozen --inexact --no-install-project \
    --extra test --extra glass --extra workflow

# Install sp_validation itself (editable) into the same venv; deps are already
# satisfied by the sync above.
COPY . /sp_validation
RUN uv pip install --no-deps -e .
