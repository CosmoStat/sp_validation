# PRD — Fork packaging, install identity, CI, and pip-only docs

Repo: `UNIONS-WL/Smokescreen`.

## Purpose

The forked Smokescreen package must be installable by downstream projects as a first-class core dependency with correct dependency metadata, its CI must gate on a theory backend that requires no heavyweight optional stack, and its install path must be documented for pip-only consumers. This PR settles the fork's install identity, rewrites its dependency declarations so a clean install pulls exactly the blinding closure — and nothing test-only into runtime — re-points CI onto a backend smoke-test, and updates the README/docs to the pip-only install form. It is the packaging-and-docs slice of the fork work and the piece other repositories depend on to declare the fork.

## Prerequisite

This PR presupposes the theory-backend refactor already in the fork: the `theory_fn(cosmo_params) -> np.ndarray` protocol is the theory entry point, the fork ships a default CCL backend (`pyccl` imported inside the backend module, not at top level), `import smokescreen` succeeds without importing any theory backend, and both a synthetic-callable and a default-CCL blind path exist. Firecrown is retained only as an inherited, lazy-imported, unsupported path — never imported at module level. Without that refactor, `smokescreen/datavector.py` imports firecrown at module load and a clean install `ImportError`s at `import smokescreen`, so the CI gate below cannot pass. This PR is separable in *review* (packaging, CI, and docs are self-contained) but its CI gate goes green only once the protocol refactor is merged; that ordering is a hard dependency, not a choice.

## Desired end state

- The fork has a single, documented install identity that a downstream `pyproject`/lockfile pins without ambiguity: a git+https URL at a pinned tag.
- A clean install of the fork (into an empty venv — no packages preinstalled) succeeds and exposes `import smokescreen`. The install pulls `pyccl` as a declared runtime dependency that the fork **uses**: the default CCL backend module imports it (at backend construction, not at `import smokescreen`). "Clean" here means the consumer preinstalls nothing; pyccl arrives as part of the fork's own runtime closure. The load-bearing property is that `import smokescreen` imports no theory backend — pyccl is imported only when the default backend is built, and firecrown only along its lazy, unsupported path — so top-level import is self-contained.
- The fork's runtime dependency set is exactly the blinding closure: no test tooling (`pytest`, `pytest-cov`) appears as a runtime `Requires-Dist`; such tooling lives in a `[test]` extra.
- `cryptography` and `sacc>=0.12` are declared runtime dependencies, and the pyproject rewrite preserves them (asserted by the acceptance criteria below, not merely assumed).
- The pyproject carries `requires-python = ">=3.12"`, no `numpy` upper bound, and retains the stock `numpy>=2.2.0` floor.
- CI's required gate builds the fork on a clean environment, asserts `import smokescreen`, and blinds a synthetic data vector through a non-CCL callable backend end-to-end.
- The fork README/docs document the pip-only install path — `pip install git+https://github.com/UNIONS-WL/Smokescreen@<tag>` — with no conda and no PyPI publish.

## Interfaces and contracts

### Install identity: pinned git+https at a tag

Downstreams depend on the fork via

```
smokescreen @ git+https://github.com/UNIONS-WL/Smokescreen@<tag>
```

`[project].name` stays `"Smokescreen"` (the import package `smokescreen` is unchanged); nothing is published to PyPI and there is no rename.

**Tag-cutting order.** The tag is the last step, not part of this PR's merge commit. It is cut only once *both* prerequisites are on the fork's default branch: the theory-backend protocol refactor (so `import smokescreen` succeeds on a clean install) and this packaging PR (so `Requires-Dist` declares `pyccl` and the pip-only docs are in place). Whichever of the two merges second, the tag is placed on the resulting default-branch commit. Concretely: do not cut the tag in this PR — merge this PR, ensure the protocol PR is also merged, then tag the merged tip. That tag is the value the downstream deps PR pins verbatim, so the two documents agree by construction. Keeping the fork diff small and clean preserves the option of PRing changes back upstream later.

### Dependency metadata (`pyproject.toml`)

The fork's `[project].dependencies` is **exactly** this set — no more, no fewer:

```
astropy>=5.2.0
cryptography
jsonargparse[signatures]>=4.0
numpy>=2.2.0
pyccl
sacc>=0.12
scipy>=1.9.0
```

Provenance (so the set is checkable, not asserted): stock Smokescreen's built metadata declares `Requires-Dist` = `astropy`, `cryptography`, `jsonargparse` (with the `[signatures]` extra), `numpy>=2.2`, `sacc>=0.12`, `scipy`. The fork keeps that set verbatim — floors and all — and adds exactly one entry, `pyccl`, which stock leaves undeclared. Nothing is dropped. Verify against the stock metadata (Smokescreen 1.5.6, read from the installed package or PyPI) before landing the rewrite; the "exactly this set" contract is stock-`Requires-Dist` **+ `pyccl`**.

Rules:
- `pyccl` is the one addition over stock: the fork's default CCL backend imports and uses it, and stock relied on it being present in a conda theory stack without declaring it. Declaring it makes a clean pip install of the fork's default blinding path self-contained. (The required CI gate blinds through a synthetic callable so it stays fast and stack-light; a separate non-required job may exercise the default CCL backend where pyccl is available.)
- `astropy`, `jsonargparse[signatures]`, `scipy` are inherited from stock `Requires-Dist` unchanged — retained, not curated. Do not promote any of them to a CLI-only or theory-only extra in this PR; that is out of scope, and the fork's own modules import them at runtime.
- `pytest`, `pytest-cov`, and any other test tooling are **not** runtime `Requires-Dist`. They move to a `[project.optional-dependencies].test` extra. A clean `pip install` of the fork must not drag pytest into a downstream runtime.
- `cryptography` and `sacc>=0.12` are declared runtime dependencies; the rewrite preserves them.
- No `numpy` **upper** bound anywhere. The `numpy>=2.2.0` **floor** is inherited from stock Smokescreen's metadata and retained unchanged; it is a floor, not the banned ceiling.
- `requires-python = ">=3.12"`.

### Fork CI (`.github/workflows/`)

One required job (`backend-smoke`):
- Fresh runner, no conda; install the fork from its own `pyproject` (the same identity path chosen above).
- Assert `python -c "import smokescreen"` succeeds.
- Run a smoke test with a concrete, observable assertion (a fresh engineer must not have to invent the fixture or tolerance):
  - **Fixture:** a small in-memory SACC vector — a single tracer pair, ~10 data points is sufficient (no real survey data, no files on disk).
  - **Backend:** a synthetic theory callable `theory_fn(cosmo_params) -> np.ndarray` returning an array aligned to the SACC rows (e.g. a smooth analytic function of the fixture's angular bins) — no CCL.
  - **Changed:** the blinded vector differs from the input — `not np.allclose(blinded, original)`.
  - **Reproducible from seed:** two blind calls with the *same* seed produce the identical shifted array — `np.array_equal(blind(seed), blind(seed))`; a *different* seed produces a different array — `not np.array_equal(blind(seed_a), blind(seed_b))`.
- Runs with `numpy` unconstrained above the `2.2` floor (no upper bound in the CI environment either).

### Fork docs (README)

The README documents the pip-only install path as the single supported form:

```
pip install git+https://github.com/UNIONS-WL/Smokescreen@<tag>
```

- No conda-forge install instructions; no PyPI `pip install smokescreen`.
- The `<tag>` in the docs is byte-identical to the tag this PR cuts and to the pin the downstream deps PR consumes.
- Any stock install instructions that assumed a conda-forge theory stack are removed or replaced by the pip-only form.
- The docs state plainly, in neutral tone, that the firecrown integration path is inherited from upstream and **unsupported** in this fork: not installed, not tested, not maintained. The default and supported theory path is the built-in CCL backend.

## Acceptance criteria

- [ ] `pip install`ing the fork into an empty venv (nothing preinstalled) succeeds and `import smokescreen` works. The install resolves its full declared closure — including `pyccl` — and the default CCL backend constructs successfully; `import smokescreen` itself imports neither pyccl nor firecrown at module load.
- [ ] The fork's built metadata (`Requires-Dist`) is **exactly** `astropy>=5.2.0`, `cryptography`, `jsonargparse[signatures]>=4.0`, `numpy>=2.2.0`, `pyccl`, `sacc>=0.12`, `scipy>=1.9.0` — and no others: no `pytest`/`pytest-cov` in runtime position.
- [ ] Fork `pyproject` carries `requires-python = ">=3.12"`, retains the `numpy>=2.2.0` floor, and carries no numpy upper bound.
- [ ] A `[project.optional-dependencies].test` extra **exists** and carries the test tooling (`pytest`, `pytest-cov`), so `pip install <fork>[test]` provisions the fork's own test run; the tooling is present there and absent from the runtime `Requires-Dist`.
- [ ] The fork README documents the pip-only install form `pip install git+https://github.com/UNIONS-WL/Smokescreen@<tag>`, with no conda or PyPI path, and the `<tag>` string is byte-identical to the pin the downstream deps PR consumes.
- [ ] Required CI job `backend-smoke` passes: `import smokescreen` plus a synthetic-callable end-to-end blind on an in-memory SACC fixture, asserting `not np.allclose(blinded, original)` (the vector changed), `np.array_equal(blind(seed), blind(seed))` (same seed reproduces), and `not np.array_equal(blind(seed_a), blind(seed_b))` (different seeds diverge).

## Non-goals

- The theory-backend protocol refactor (`theory_fn`) and the fixed pure draw — landed separately; this PR is a hard consumer of the protocol (see Prerequisite) and defines only packaging, CI, and docs.
- Any change to `UNIONS-WL/Smokescreen` blinding semantics, encryption, or the shift math.
- Downstream (sp_validation) dependency declaration, container, or lockfile changes — those live in the sp_validation deps PR, which this PR unblocks but does not touch.
- Publishing to PyPI or conda, renaming the distribution, or opening any PR against DESC upstream `LSSTDESC/Smokescreen`.
