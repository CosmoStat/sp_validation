---
id: 01KX8781PRX0720TKFAEXY7RMY
name: 'sp_validation: declare cosmo_numba + migrate to uv/uv.lock'
status: active
tags:
    - sp_validation
    - deps
    - future
    - constitution
created-at: 2026-07-11T11:13:29.048895365+02:00
updated-at: 2026-07-11T12:55:32.54099491+02:00
outcome: 'PR #266 open (Closes #265), CI image build running — the validator of desired-state #4. Change: declare cosmo-numba @ aguinot/cosmo-numba@main (main carries the numpy-2 FFT fix via rocket-fft AND declares its deps, so numba''s window reaches the resolver) + pin numba directly as insurance; commit uv.lock (numpy 2.4.6, cs_util 0.2.2); Dockerfile installs from the lock via `uv sync --frozen --inexact` into the base /app/.venv, dropping the ad-hoc snakemake and cs_util `--upgrade` layers. Next: confirm the image build goes green (then it''s ready for Cail to merge — merge timing his). Open: cv_runner + unions_wl left undeclared (no resolvable source); numba insurance pin is droppable; shapepipe cs_util refresh now optional (we''re decoupled from the base''s cs_util).'
shuttle:
    kind: oneshot
    host: candide
    agent: claude-opus
    effort: medium
    runtime:
        dispatched_at: "2026-07-11T09:41:21.794114Z"
        session_uuid: 56c5f57c-71f7-43e0-8f81-08490705c2e6
---

sp_validation's dependency environment is un-locked and incomplete, and it just cost a day: the published image (built `FROM` the ShapePipe image, which carries a numba-compatible numpy via its `uv.lock`) drifted numpy to 2.5 during its own unpinned install layers, killing numba — and with it the ShapePipe ngmix stage the image is supposed to be able to run. Root cause is structural, not a version accident: `cosmo_numba` (the numba-bearing dependency, imported in `b_modes.py`, `importorskip`'d in tests) is not declared in `pyproject.toml`, so the resolver never sees numba's numpy window; and with no lockfile, every image build re-resolves from scratch. This fiber makes sp_validation's environment reproducible the way [[science/unions/shapepipe|shapepipe]]'s already is, and thereby restores the one-container premise of the image-sims chain ([[science/unions/shapepipe-program/image-sims-workflow]] — its `.smk` currently carries a two-image workaround, `sif_pipeline`, to be collapsed once this lands).

## Desired State

1. **`cosmo_numba` is a declared dependency** of sp_validation (git ref or PyPI, whichever CosmoStat/cosmo-numba supports), so numba's constraints reach the resolver. Any other imported-but-undeclared deps found during the audit get declared too (sweep `src/` and `scripts/` imports against `pyproject.toml`).
2. **sp_validation is uv-managed with a committed `uv.lock`**, mirroring shapepipe's setup: `pyproject.toml` holds abstract minimums, `uv.lock` the exact pins, never hand-edited. CI and the Dockerfile install from the lock.
3. **The Dockerfile has no unscoped `--upgrade`** — the cs_util layer uses `--upgrade-package cs_util` (or the lock makes the layer unnecessary). No install layer can silently move a base-image pin.
4. **The rebuilt image runs the full ShapePipe chain**: `python -c "import numba, cosmo_numba"` passes and numpy inside the image satisfies numba's window; verified by running the image-sims `im_pipeline` stage (or at minimum a `shapepipe_run` ngmix smoke) in the new image. This is the checkable condition that lets the image-sims `.smk` drop `sif_pipeline`.
5. **The work lands as a PR** against sp_validation's integration branch, on its own feature branch. Pushing that feature branch (needed for CI image builds) is in scope; pushing/merging any shared branch is not — merge timing is Cail's.

## Boundaries

- Don't hand-pin numpy ceilings to track numba's compatibility — that's exactly the vigilance-based fix this task replaces with resolver-visible constraints.
- Don't touch the image-sims workflow rules in this fiber; the `sif_pipeline` collapse happens in the image-sims fiber once the healthy image is published and verified.
- Coordinate-lightly rule: Martin builds from the same Dockerfile lineage — keep the diff minimal and legible, no drive-by refactors of his layers.

## Status

Landed as **PR #266** (Closes #265, base `develop`, branch `uv-lock-cosmo-numba`), committed at `95843df`. All of desired-state #1–#3 and #5 are done; #4 is being validated by the CI image build triggered on push.

**The resolution of the cosmo_numba question** (the session's main find): the constitution's premise — "declare cosmo_numba → numba's numpy window reaches the resolver" — was only half right. cosmo-numba is **not on PyPI**. Its refs diverge: the `fix/numpy2-fft-compat` fork tip (`db452c48`) declares `dependencies = []` (empty), so it contributes *nothing* to the resolver; `aguinot/cosmo-numba@main` declares its deps from `requirements.txt` (which uv reads fine) **and** carries the real numpy-2 FFT fix as a `rocket-fft` dependency (Cail's own PR-#17 objmode workaround was closed in favor of this — see his comment there). So we point at **@main**. Belt-and-suspenders: `numba` is *also* pinned directly in sp_validation, because cosmo-numba's dep metadata proved it can silently empty out between refs. Result: numpy locks to 2.4.6 (inside numba 0.66's <2.5 window).

**Install model:** sp_validation builds `FROM` shapepipe's image, which provides a uv venv at `/app/.venv` (VIRTUAL_ENV). The Dockerfile does `uv sync --frozen --inexact` with `UV_PROJECT_ENVIRONMENT=/app/.venv` — `--inexact` keeps the inherited ShapePipe stack (not in our lock) rather than pruning it. Because our lock pins cs_util 0.2.2, we're **decoupled from the base image's cs_util** — the old `--upgrade` layer is gone and the shapepipe-side cs_util refresh Cail floated is now optional housekeeping, not a blocker here.

**Next session picks up at:** watch PR #266's image build. Green → desired-state #4 met (import smoke + fast unit suite pass in the rebuilt image), ready for Cail to merge (his call). Red → read the failing step: likely candidates are NumbaQuadpack's git build (needs the apt build tools, present) or a base-vs-lock version clash surfaced by `uv sync --inexact` moving numpy/scipy under the inherited ngmix. Do **not** close this fiber until the build is green and merged; merge timing is Cail's.

**Open threads:** `cv_runner` (imported in `workflow/scripts/cv_*.py`) and `unions_wl` (one footprint script) are left undeclared — no resolvable source found; Cail flagged cv_runner as not his. The `numba` insurance pin is deliberate but droppable once trust in cosmo-numba@main's metadata is established.
