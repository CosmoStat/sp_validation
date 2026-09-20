# Contributing

Thanks for your interest in `sp_validation`! This guide covers how to set up a
development environment, run the tests, and propose changes.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Development environment

`sp_validation` depends on a large scientific stack (`treecorr`, `pyccl`,
`pymaster`, `healpy`, …). The simplest, most reproducible way to develop is
inside the project container, which ships the full stack pre-built.

### Container (recommended)

CI builds and pushes an image on **every** push, tagged by the sanitized branch
name (see
[`.github/workflows/deploy-image.yml`](.github/workflows/deploy-image.yml)), so
`:develop` tracks the integration branch and your branch has an image of its
own. Nothing is built by hand.

You keep your own copy of the image. `spv-container` — stdlib-only, so it runs
straight from a checkout: symlink it onto your PATH (`ln -s
"$PWD/src/sp_validation/container.py" ~/.local/bin/spv-container`, the README's
install step) or call it as `python3 src/sp_validation/container.py` — pulls it to
`~/.cache/sp_validation/sp_validation.sif` and runs things inside it:

```bash
spv-container pull       # fetch :develop; ~1.5 GB, so do it from a compute node
spv-container status     # which layer is live, and how current it is
spv-container exec bash  # an interactive shell inside it
```

That image is read-only. When you need a package it does not carry yet, unpack a
writable sandbox once with `spv-container sandbox` and install into it with
`spv-container exec --writable pip install <pkg>`; the sandbox then takes
precedence everywhere, workflow jobs included. Treat it as an exploration tool —
the real fix is adding the dependency to `pyproject.toml` — and reset it with
`spv-container pull && spv-container sandbox --force`.

Analysis runs through Snakemake, which wraps every job in `apptainer exec`
against that same image for you — see
[`workflow/README.md`](workflow/README.md) for the profiles and the details.

Two things worth knowing while developing:

- **Your checkout's code is what runs.** The workflow prepends the launched
  checkout's `src/` to the container's `PYTHONPATH`, so the image supplies the
  dependency stack and your working tree supplies `sp_validation`. No rebuild
  needed to test a change. (Caveat: `rerun-triggers: code` does not watch
  `src/`, so force reruns after editing a module.)
- **To test a branch's own image** — when the *stack* changed, not just `src/` —
  point the workflow at its CI tag:

  ```bash
  snakemake --profile workflow/profiles/candide \
      --config container=docker://ghcr.io/cosmostat/sp_validation:my-branch <target>
  ```

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

**`develop` is the gate — and on a PR it fixes for you.** On every push to
`develop` and every PR into it, CI runs the full ruff policy.

- **On a PR from a branch in this repo** → the gate doesn't just report, it
  **fixes**: it runs `ruff format` + `ruff check --fix` and **pushes the result
  back to your branch as the `github-actions` bot**, then re-checks. If that
  cleaned everything, the PR comment goes green (`🤖 autofix pushed …, ruff is
  clean`) and there's nothing to do — just `git pull` to pick up the commit. If
  anything ruff *won't* safely fix survives (undefined names, unused variables,
  other judgement calls), the check stays **red** and the comment lists **only
  the residual** — the mechanical stuff is already handled. So you rarely touch
  ruff by hand; when you do, it's the real judgement calls.
- **On a PR from a fork** (where CI can't push to your branch) → it posts (and
  keeps updating) a **comment on the PR** with the full violation list. Push a
  fix and the comment turns green.
- **On a direct push to `develop`** (no PR) → it opens (or updates) a single
  **lint-debt issue assigned to you**, which auto-closes when CI is green.

So: warn while you work, and for same-repo PRs the gate mostly cleans up after
you before it lands.

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
