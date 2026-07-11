# PRD — Dependencies

Repo: `CosmoStat/sp_validation`.

## Purpose

Blinding the tomographic data vector requires Smokescreen and its CCL theory backend. This PR makes those dependencies installable and reproducible as **core** dependencies: it pins the `UNIONS-WL/Smokescreen` fork by git tag or SHA, promotes `sacc` to a top-level dependency, adds `cryptography`, raises the package's Python floor from `>=3.11` to `>=3.12`, and installs the closure in the container. There is no optional blinding extra — with firecrown out of the install closure the blinding closure is small enough (fork + `pyccl` + `cryptography`, with `sacc`) to live in the core install.

## Prerequisite (blocking)

This PR consumes a **single fork install identity** — one `git+https` reference into `UNIONS-WL/Smokescreen`, pinned by tag or SHA, produced by the fork's packaging PR and pinned here verbatim:

```
smokescreen @ git+https://github.com/UNIONS-WL/Smokescreen@<ref>
```

`<ref>` is a tag or commit SHA that (a) declares `pyccl` in the fork's `Requires-Dist` and (b) has made the firecrown import lazy (no module-level firecrown import) so a clean install does not require firecrown. Pinning any earlier fork commit fails: stock Smokescreen imports `firecrown` and `pyccl` at module level (`datavector.py`) yet declares neither, so any install lacking firecrown `ImportError`s at `import smokescreen`. Nothing is published to PyPI — the fork ships as a git pin only.

**Cold-implementability hole:** `<ref>` is the one load-bearing string this PR pins, and it does not exist until the fork's packaging PR lands. Until it is supplied, this PR cannot be implemented and the acceptance criteria below cannot be run — a reviewer must substitute the real tag/SHA before any install-resolution check is executable. This is a genuine cross-PR dependency, not an unfilled value: **this PR is blocked on the fork's packaging PR.**

## Support-matrix change

The current package declares `requires-python = ">=3.11"` (ruff `target-version = "py311"`). Smokescreen requires `>=3.12`, so this PR raises sp_validation's floor to `>=3.12`. This **drops Python 3.11 support** for sp_validation as a whole — a genuine support-matrix decision, not bookkeeping. It is safe against the container base (`python:3.12-slim-bookworm`, via `shapepipe:develop`) and against CI (which runs only inside that image); no external sp_validation consumer or CI matrix is known to run 3.11.

## Desired end state

- The blinding dependencies are **core** runtime dependencies in `[project].dependencies`: the fork (via the prerequisite ref), `cryptography`, and `sacc>=0.12`. `pyccl` is already core.
- There is **no `[blinding]` optional-dependency extra.** It is deleted; blinding installs with the base package.
- `requires-python` is `>=3.12`; ruff `target-version` is `py312`.
- **No `firecrown` anywhere** — no dependency, no `--no-deps` install step, no `scripts/patch_firecrown.py`, no firecrown import smoke-check, no guarded firecrown CI job. **No `numpy` upper bound anywhere** in the packaging.
- The container installs the full core closure, captured in `uv.lock` and the `Dockerfile`.
- The CI deploy image builds cleanly and runs the full fast test suite. `deploy-image.yml` already runs an `import sp_validation` smoke test and the fast suite in-image; this PR adds one new step to that job — an `import smokescreen` check that resolves the fork install.

## Interfaces and contracts

### `pyproject.toml`

Raise the floor:

```toml
[project]
requires-python = ">=3.12"
```

Add the blinding closure to core `dependencies` (`pyccl` is already declared there):

```toml
dependencies = [
    ...
    "smokescreen @ git+https://github.com/UNIONS-WL/Smokescreen@<ref>",
    "sacc>=0.12",
    "cryptography",
    ...
]
```

- `<ref>` is the value fixed by the prerequisite above, copied byte-for-byte from the fork's packaging PR.
- `sacc>=0.12` and `cryptography` are declared explicitly here so the core runtime closure is self-documenting and independent of fork-metadata drift. Do not pin either more tightly than the fork does.

There is no `[project.optional-dependencies].blinding` table.

### Ruff target

```toml
[tool.ruff]
target-version = "py312"
```

### `Dockerfile`

The editable install line currently reads:

```dockerfile
RUN uv pip install --no-cache-dir -e '.[test,glass]'
```

No extra is added — the blinding closure is core, so the existing line installs it unchanged. Only the resolved `uv.lock` (below) changes to point Smokescreen at the fork ref. The base image already resolves to Python 3.12, so the raised floor needs no base-image change.

### `uv.lock`

Regenerate so the resolved Smokescreen entry points at the fork ref and the lockfile carries the closure (`pyccl`, `cryptography`, `sacc`). No `firecrown` or `numpy<2.x` marker may appear in the resolved graph.

## Acceptance criteria

These criteria are runnable only once the prerequisite `<ref>` is supplied and substituted for the placeholder (see Prerequisite); an earlier fork pin fails by dragging firecrown in.

- `uv pip install -e .` on Python 3.12 resolves and installs the blinding closure with no firecrown and no numpy ceiling in the resolved set.
- `python -c "import smokescreen"` succeeds in the installed environment without firecrown present.
- `grep -n 'requires-python' pyproject.toml` shows `>=3.12`; ruff `target-version` is `py312`.
- Core `dependencies` contains the fork git ref pin, `sacc>=0.12`, and `cryptography` (plus the pre-existing `pyccl`); there is **no `[blinding]` extra** in `pyproject.toml`.
- `grep -rn firecrown pyproject.toml Dockerfile uv.lock scripts/` returns nothing; `scripts/patch_firecrown.py` does not exist.
- `grep -rn 'numpy<' pyproject.toml uv.lock` returns nothing (no ceiling).
- The deploy-image CI job (`.github/workflows/deploy-image.yml`) builds the image and passes the pre-existing `import sp_validation` smoke test, the new `import smokescreen` check this PR adds, and the fast test suite (`pytest -m "not slow"`) against the freshly built image before publishing.

## Non-goals

- Wiring Smokescreen into the blinding pipeline, drawing shifts, or computing theory — this PR only makes the dependencies installable.
- Any fork-side change. This PR produces nothing in `UNIONS-WL/Smokescreen` and consumes its settled install identity as a blocking prerequisite.
- Supporting a firecrown theory backend. sp_validation never installs firecrown; the only theory path here is CCL, and no numpy ceiling is imposed. (The fork retains an inherited, lazy, unsupported firecrown path that this install never pulls in — firecrown stays undeclared and uninstalled.)
- Adopting Smokescreen against DESC upstream, or publishing to PyPI. The pin targets the `UNIONS-WL/Smokescreen` fork by git tag.
