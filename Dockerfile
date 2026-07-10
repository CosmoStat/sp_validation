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

# Install with the test + glass extras so the image can run the unit suite in CI
# *and* the GLASS map-level mock test. `glass` (Generator for Large Scale
# Structure) ships `glass.ext.camb`; `cosmology` provides the `Cosmology` wrapper
# (`Cosmology.from_camb`) GLASS consumes. Both come in via the `[glass]` extra.
RUN uv pip install --no-cache-dir -e '.[test,glass]'
