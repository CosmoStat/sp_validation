---
id: 01KTT53Y3681X6FTJ6WM4YZRMV
name: 'CI hermeticity: guards in the git-less image'
status: closed
tags:
    - sp-validation
    - finding
created-at: 2026-06-11T03:35:31.430313054+02:00
outcome: 'All 7 in-image test failures after the phase-2 moves were environmental, none were real stale references: the Docker image is built from the Git build context (tracked content only, no .git), so `git ls-files` exits 128 there — the dangling-reference and tracked-symlink guards now fall back to a tree walk, which in the image scans exactly the tracked set; the bmodes dry-run guard is candide-bound by design (candide-absolute catalog configfile + cluster data) and skips off-cluster like the test_cosmo_val data guards, while satisfying the Snakefile''s `envvars: PYTHONUNBUFFERED` itself. Separately, the docs workflow was red because sphinxawesome-theme 6.0.3 on PyPI is a broken wheel (dist-info only, no python module) — pinned `!=6.0.3` in the docs extra.'
horizon: now
---

The back-pressure suite must be green in two environments with different
furniture, and each guard has to know which differences are signal:

- **Docker image** (`docker run … pytest`): tracked content only, no `.git`
  directory, no cluster mounts. Git-querying guards
  (`test_dangling_move_references`, `test_tracked_symlinks`) fall back to a
  full tree walk when `git ls-files` fails — equivalent there, since the image
  *is* the tracked set — so the guards keep their teeth in CI rather than
  skipping.
- **candide checkout**: full data landscape, untracked run-dir furniture.
  Guards that genuinely need it (`test_bmodes_workflow_dry_run`, the
  config-path and catalog guards) carry the off-cluster skip; the dry-run
  guard also exports `PYTHONUNBUFFERED=1` itself so the Snakefile's `envvars:`
  declaration doesn't leak a dependency on the invoking shell.

A separate trap discovered the same night: a `pull_request`-triggered workflow
(API docs) silently stops running when GitHub marks the PR conflicting — no
red check, just absence. A clean merge from `develop` restored mergeability
and the docs check. And `uv pip install` reporting success is not proof a
package is importable: sphinxawesome-theme 6.0.3 installs cleanly and ships
zero code.
