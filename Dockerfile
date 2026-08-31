# Development image with more bells and whistles
FROM ghcr.io/cosmostat/shapepipe:im_sims

# liblapack-dev: cosmosis's MultiNest links -llapack, and the base image ships
# only the runtime liblapack.so.3 (no dev symlink). The gsl/cfitsio/fftw3 dev
# packages are what the CosmoSIS Standard Library's C sources compile against
# (they are the headers CSL's own CI installs); git is for cloning it.
RUN apt-get update -y --quiet --fix-missing && \
    apt-get dist-upgrade -y --quiet --fix-missing && \
    apt-get install -y --quiet \
        autoconf \
        automake \
        libtool \
        pkg-config \
        git \
        htop \
        npm \
        tmux \
        liblapack-dev \
        libgsl-dev \
        libcfitsio-dev \
        libfftw3-dev

# The base shapepipe image provides a uv-managed venv at /app/.venv (exported as
# VIRTUAL_ENV); install sp_validation's deps into that same venv rather than
# spawning a second one under /sp_validation.
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
# $HOME is bind-mounted under apptainer, so uv would otherwise discover the
# host's managed CPythons -- including newer ones that satisfy requires-python
# -- and build a venv against an interpreter carrying none of this stack.
ENV UV_PYTHON=/app/.venv/bin/python \
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

# The CosmoSIS Standard Library: the module files (camb interface, projection,
# 2pt likelihood, ...) the cosmo_inference pipelines name. The `workflow` extra
# above installs cosmosis itself; CSL is a separate tree of modules that is not
# on PyPI and has to be built against that install, so it is cloned and compiled
# here rather than left to each user (which is what the .ini templates used to
# assume, hard-coding one person's home directory).
#
# CSL_REF pins UNIONS-WL fork main: Sacha's four UNIONS commits reapplied
# on current upstream, including the scipy lpn fix.
ARG CSL_REPO=https://github.com/UNIONS-WL/cosmosis-standard-library.git
ARG CSL_REF=b7b1552a02ad9c39c9bb1e68e3f17213a8f740e1
ENV CSL_DIR=/opt/cosmosis-standard-library

# `python -m cosmosis.configure` emits the exports (COSMOSIS_SRC_DIR et al.)
# every CSL Makefile includes its compiler config from. Evaluated directly
# rather than through the `cosmosis-configure` wrapper: that wrapper's
# am-I-sourced probe reads unset zsh/ksh variables, which `set -u` turns into
# an error, and its `exit` then ends the sourcing shell with status 0 — make
# never runs and the layer still "succeeds". The trailing `test -f` keeps any
# such silent no-op loud.
#
# `make -C shear` rather than a bare `make`: the top-level target also descends
# into likelihood/, which builds the Planck, WMAP and ACT likelihoods -- large,
# data-dependent, and unused by any UNIONS pipeline. Everything our .ini
# templates reference is either pure Python (consistency, sample_S8, camb,
# load_nz_fits, photoz_bias, linear_alignment, add_intrinsic, shear_m_bias,
# xi_sys, 2pt_like -- no Makefile in those trees at all) or lives under shear/:
# `limber`, which project_2d.py links, and `cl_to_xi_nicaea`, whose
# nicaea_interface.so the 2pt_shear stage loads.
RUN bash -c 'set -eo pipefail; \
    export PATH=/app/.venv/bin:$PATH; \
    git clone --filter=blob:none "$CSL_REPO" "$CSL_DIR"; \
    cd "$CSL_DIR"; \
    git checkout --detach "$CSL_REF"; \
    cmds=$(python -m cosmosis.configure); \
    eval "$cmds"; \
    export GSL_INC=/usr/include GSL_LIB=/usr/lib/x86_64-linux-gnu; \
    make -C shear; \
    test -f shear/cl_to_xi_nicaea/nicaea_interface.so'

# Install sp_validation itself (editable) into the same venv; deps are already
# satisfied by the sync above.
COPY . /sp_validation
RUN uv pip install --no-deps -e .
