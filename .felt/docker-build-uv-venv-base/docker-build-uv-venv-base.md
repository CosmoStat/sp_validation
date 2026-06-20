---
id: 01KTCHX02B8APEX03YEC97AW4E
name: 'Docker build: uv-venv base image'
status: closed
tags:
    - docker
    - ci
created-at: 2026-06-03T10:22:53.245547768+02:00
closed-at: 2026-06-10T17:13:38.572339006+02:00
outcome: 'sp_validation image fixed for the uv-venv shapepipe:develop base: autotools restored via apt for pymaster, uv pip install so packages land in /app/.venv. Standing risk documented: moving base tag can break again — diff base env assumptions before blaming our code.'
---

The `Dockerfile` builds `FROM ghcr.io/cosmostat/shapepipe:develop` — a **moving tag**.
A rebuild of that upstream image on 2026-05-31 changed the environment underneath us, and
the GitHub Actions "Create and publish a Docker image" workflow went red. The same
`sp_validation` code built green on 2026-05-04. Two independent breakages, fixed in two
commits on `develop`:

**1. pymaster can't compile its C library** (commit `a1e2a41`). pymaster 2.6 builds
`libnmt` from sdist via autotools; the new base image dropped the `libtool` package, so
`autoreconf` failed with `Makefile.am:3: error: Libtool library used but 'LIBTOOL' is
undefined` → `pip install -e .` aborted. Fix: add `autoconf automake libtool pkg-config`
to the apt step. (The base still ships GSL/FFTW/CFITSIO dev libs, so libtool was the only
missing piece — but pymaster compiling from source at all means those could vanish next.)

**2. `import sp_validation` fails at runtime even though install succeeded** (commit
`d4a64e4`). The new base image runs out of a **uv venv**: `VIRTUAL_ENV=/app/.venv`,
`include-system-site-packages = false`, **shapepipe itself lives in the venv**
(`/app/src/shapepipe`), and the venv has **no `pip`**. So bare `pip install -e .` fell
through to `/usr/local/bin/pip` and installed into the *system* python
(`/usr/local/lib/python3.12`), invisible to the `python` that `docker run` actually uses
(the venv). The CI smoke test `docker run <img> python -c "import sp_validation"` then
threw `ModuleNotFoundError`. Fix: install with **`uv pip install`** (uv ships at
`/usr/local/bin/uv`, honors `VIRTUAL_ENV`) so packages land in `/app/.venv` alongside
shapepipe. Verified the image green after both fixes.

**Diagnosis tool:** `apptainer exec --cleanenv docker://ghcr.io/cosmostat/shapepipe:develop
bash -lc '...'` lets you inspect the base image's real env (PATH, `pyvenv.cfg`, where
shapepipe imports from) without a full Docker build — far faster than CI round-trips. Pin
host PATH (`export PATH=/usr/local/bin:/usr/bin:/bin`) so host tools don't leak in.

**Standing risk:** because the base is a moving tag, a future upstream rebuild can break the
build again with no change on our side. If this recurs, first `diff` the base image's env
against what the Dockerfile assumes (uv venv, autotools, dev libs) before touching our code.

## CI now runs the unit suite inside this image

As of PR #186 (2026-06-03), `deploy-image.yml` runs `pytest -m "not slow"` inside the built
image, right after the `import sp_validation` smoke test. This is deliberate: the heavy
system-dependency stack (pymaster's GSL/FFTW/CFITSIO, etc.) already lives in the image, so
running tests there is reliable — re-solving those deps on a bare GitHub runner would be the
same fight that broke the build above. The Dockerfile installs the `[test]` extra (`uv pip
install -e '.[test]'`) so pytest is present. The dead `ci-build.yml` (master-triggered,
py3.8, conda, `setup.py test` — never fired) was deleted; `cd-build.yml` is the same era and
still dead (docs→gh-pages), left for a separate cleanup. Tests must stay hermetic for the
`not slow` set: catalog-data-dependent tests are marked `@pytest.mark.slow` so they only run
on the cluster (the image has `cat_config.yaml` but no catalogs).
