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

# The base image installs into the system interpreter (/usr/local); use `uv pip`
# so the heavy scientific stack and our deps land where `python` resolves.
RUN uv pip install --no-cache-dir \
    snakemake

# The base shapepipe image ships cs_util 0.1.9, and `uv pip install -e` does NOT
# upgrade an already-satisfied dependency to meet a *new* lower bound (astral-sh/uv
# #8410). sp_validation now needs `cs_util.size` (cs_util>=0.2.1), so upgrade it
# explicitly here — otherwise the editable install silently keeps 0.1.9 and the
# galaxy import smoke test fails. shear_psf_leakage@develop allows cs-util<0.3,
# so 0.2.x satisfies the whole graph.
RUN uv pip install --no-cache-dir --upgrade 'cs_util>=0.2.1'

WORKDIR /sp_validation
COPY . /sp_validation

# Install with the test + glass + blinding extras so the image can run the unit
# suite in CI, the GLASS map-level mock test, *and* the SACC/Smokescreen blinding
# stack. `glass` (Generator for Large Scale Structure) ships `glass.ext.camb`;
# `cosmology` provides the `Cosmology` wrapper (`Cosmology.from_camb`) GLASS
# consumes. The `[blinding]` extra (firecrown + smokescreen) needs the override
# file: firecrown declares conda-forge-only / unused sampler connectors as hard
# deps — see uv-overrides.txt for the full story.
RUN uv pip install --no-cache-dir --overrides uv-overrides.txt -e '.[test,glass,blinding]'

# Same uv gotcha as the cs_util upgrade above (astral-sh/uv #8410): if the base
# image already carries a numpy that violates the [blinding] extra's new
# `numpy<2.5` cap (firecrown 1.15.1 breaks on numpy 2.5 at import), the
# editable install won't move it. Request the bound explicitly so the image is
# deterministic either way; numpy 2.4.x is ABI-compatible with the compiled
# stack (verified: pyccl/camb/treecorr/healpy/pymaster + fast suite).
RUN uv pip install --no-cache-dir 'numpy>=2.2,<2.5'

# firecrown is distributed for conda-forge (where NumCosmo always exists) and
# hits NumCosmo at import time in a pip env, on paths unrelated to our use.
# This patches the installed tree (surgical, pinned-version-checked, loud on
# mismatch) and verifies `import firecrown.likelihood; import smokescreen`.
RUN python scripts/patch_firecrown.py
