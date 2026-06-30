# Contributing

Thanks for your interest in `sp_validation`! This guide covers how to set up a
development environment, run the tests, and propose changes.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Development environment

`sp_validation` depends on a large scientific stack (`treecorr`, `pyccl`,
`pymaster`, `healpy`, …). The simplest, most reproducible way to develop is
inside the project container, which ships the full stack pre-built.

### Container (recommended)

```bash
# build a writeable sandbox from the published image
apptainer build --sandbox sp_validation docker://ghcr.io/cosmostat/sp_validation:develop
apptainer shell --writable sp_validation
```

The image is rebuilt and pushed on every push to `develop` (see
[`.github/workflows/deploy-image.yml`](.github/workflows/deploy-image.yml)), so
`:develop` always tracks the latest integration branch.

### Local install with `uv`

If you prefer a local environment, use [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[develop]'   # runtime + test + docs dependencies
```

Note that some dependencies (e.g. `pymaster`) compile C extensions and need a
toolchain (`autoconf`, `automake`, `libtool`, `pkg-config`) available.

## Running the tests

```bash
pytest                       # full suite
pytest -m "not slow"         # skip the slow tests
pytest src/sp_validation/tests/test_cosmology.py::test_name   # a single test
```

Tests live in `src/sp_validation/tests/`. The default options (configured in
`pyproject.toml`) collect from there and report coverage. CI runs this suite
inside the freshly-built container image *before* publishing it, so a failing
test blocks the image push.

## Code style and the lint gate

We use [`ruff`](https://docs.astral.sh/ruff/) for both formatting and linting
(line length 88). The policy lives in `pyproject.toml` and is region-aware:
`src/sp_validation/` is strict, while the analysis/workflow/script trees waive a
few intentional patterns (`sys.path` edits before imports, star-imports).

```bash
ruff check            # report lint issues
ruff check --fix      # auto-fix what it can
ruff format           # format the tree
```

**Local hooks auto-fix the safe stuff, warn on the rest.** The
[`pre-commit`](https://pre-commit.com/) hooks auto-apply everything ruff can fix
safely — `ruff format` plus `ruff check --fix`'s safe fixes (import sorting,
unused imports). When that rewrites a staged file the commit stops *once* so you
`git add` the result and commit again. Anything ruff *won't* safely fix —
undefined names, unused variables, other judgement calls — is printed as a
**warning** and never blocks the commit. Judgement-call lint stays out of your
way locally; the gate below is where it's enforced.

**`develop` is the gate.** On every push to `develop` and every PR into it, CI
runs the full ruff policy. If it fails, two things happen: the check goes **red
and blocks the merge**, and a bot opens (or updates) a single **lint-debt issue
assigned to you** listing the violations. Fix them and push again — the issue
**auto-closes when CI is green**. So: warn while you work, clean before it lands.

## Commit hygiene (notebooks & large files)

The repository's history is heavy from committed notebook outputs. Alongside the
warn-only ruff hooks above, two **blocking** `pre-commit` hooks guard against
more of it: `nbstripout` (strips notebook outputs on commit) and a large-file
check (2 MB). These block because heavy content is expensive to undo once it is
in history. Activate everything once per clone:

```bash
pre-commit install
```

## Proposing changes

1. Branch off `develop` (the integration branch — `master` is no longer used):
   `git checkout -b feature/my-change develop`.
2. Make your change with focused, clearly-described commits.
3. Add or update tests when you change behaviour, and run `pytest` + `ruff check`.
4. Open a pull request **into `develop`**. CI must pass (the container image
   builds and the test suite runs) before merge.

For larger or analysis-affecting changes, open an issue first so we can discuss
the approach.

## Documentation

API documentation is built with Sphinx and deployed to
[GitHub Pages](https://cosmostat.github.io/sp_validation/) from `develop`. To
build it locally:

```bash
uv pip install -e '.[docs]'
sphinx-apidoc -t docs/_templates -feTMo docs/source src/sp_validation
sphinx-build -b html docs/source docs/_build
```
