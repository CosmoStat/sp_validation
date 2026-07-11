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
updated-at: 2026-07-11T11:35:39.19956558+02:00
outcome: 'Spun off 2026-07-11 from the image-sims gate failure: sp_validation''s published image drifted numpy to 2.5, breaking numba, because cosmo_numba (its numba-bearing dep, imported in b_modes.py and importorskip''d in tests) is undeclared and there is no lockfile — the resolver can''t see numba''s numpy window. Fix: declare cosmo_numba as a dependency, adopt uv with a committed uv.lock (mirroring shapepipe), and drop unscoped ''--upgrade'' install layers from the Dockerfile (use --upgrade-package). Unblocks collapsing the image-sims chain back to one container.'
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
