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

# The [blinding] extra (SACC/Smokescreen blinding stack: firecrown + smokescreen)
# is not in uv.lock — firecrown is not on PyPI and declares conda-forge-only /
# unused sampler connectors as hard deps, so it needs the override file (see
# uv-overrides.txt for the full story). Installed as a separate pass on top of
# the locked sync.
RUN uv pip install --no-cache-dir --overrides uv-overrides.txt -e '.[blinding]'

# Same uv gotcha as the cs_util upgrade above (astral-sh/uv #8410): if the base
# image / locked sync already carries a numpy that violates the [blinding]
# extra's `numpy<2.5` cap (firecrown 1.15.1 breaks on numpy 2.5 at import), the
# editable install won't move it. Request the bound explicitly so the image is
# deterministic either way; numpy 2.4.x is ABI-compatible with the compiled
# stack (verified: pyccl/camb/treecorr/healpy/pymaster + fast suite).
RUN uv pip install --no-cache-dir 'numpy>=2.2,<2.5'

# firecrown is distributed for conda-forge (where NumCosmo always exists) and
# hits NumCosmo at import time in a pip env, on paths unrelated to our use.
# This patches the installed tree (surgical, pinned-version-checked, loud on
# mismatch) and verifies `import firecrown.likelihood; import smokescreen`.
RUN python scripts/patch_firecrown.py

# Install sp_validation itself (editable) into the same venv; deps are already
# satisfied by the sync + blinding-extra install above.
COPY . /sp_validation
RUN uv pip install --no-deps -e .
