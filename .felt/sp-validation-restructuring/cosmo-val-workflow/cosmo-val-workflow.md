---
id: 01KV0B16CX053QB9PBQDTCJSDR
name: Decompose run_cosmo_val into a Snakemake workflow
status: active
created-at: 2026-06-13T13:14:19.677661469+02:00
---

Turned the linear cosmo_val/run_cosmo_val.py driver (one in-memory `cv`, ~13
cv.<method>() calls linked by lazy properties) into a Snakemake workflow on
branch feature/cosmo-val-workflow (from cleanup/restructuring).

## Shape
- Generic layer: workflow/rules/cosmo_val.smk + one thin script per rule
  (workflow/scripts/cv_*.py) + shared harness cv_runner.py. Activated only when
  config has a `cosmo_val` block, so papers/bmodes is untouched.
- Paper layer: papers/cosmo_val/ composes the workflow module (bmodes pattern),
  default target `all` builds the whole suite.
- common.py gains cv_init_params (centralized constructor), cv_basename,
  CV_SENTINELS, CV_RUNDIR.

## Rule graph (rulegraph-verified)
rho_tau_stats, xi (EXISTING rules in twopoint.smk — cosmo_val consumes them,
does not duplicate) feed: cv_plot_rho_stats, cv_plot_tau_stats, cv_rho_tau_fits,
cv_plot_2pcf, cv_ratio_xi_sys_xi (joins rho/tau + xi), cv_pure_eb, cv_cosebis.
cv_pure_eb + cv_cosebis (+ optional cv_pseudo_cl) -> cv_summarize_bmodes.
Leaves: cv_footprints, cv_weights, cv_additive_bias. All -> rule all.

## Core tension resolution (in-memory cv vs file I/O)
Each rule re-instantiates cv; DAG is expressed through real data products, not
the shared object.
- Durable-product methods own compute rules keyed on those files (pure_eb npz,
  cosebis npz, pseudo_cl FITS).
- Pure-plot methods whose figure paths derive from internal handler state ->
  sentinel under output/snakemake_sentinels.
- Lazy state never persisted by cosmo_val.py: c1/c2 -> additive_bias.json;
  xi_psf_sys -> recomputed in cv_ratio_xi_sys_xi (cheap vs the science).
- summarize_bmodes needs live result objects (TreeCorr gg, not npz-able) so its
  rule re-runs plot_pure_eb/plot_cosebis in-process; they reload existing data
  via skip-if-exists and recompute only the cheap PTE stats — exactly what the
  original linear driver did on its shared cv.

## Status
snakemake -n all: valid 19-job DAG. Rules run in isolation. Structural guards
test_dangling_move_references + test_tracked_symlinks green;
test_config_paths_exist fails only on pre-existing worktree-isolation (gitignored
output/ dirs absent in fresh checkout), unrelated to this work.

## For Cail to decide
- versions/npatch in config are a placeholder fiducial (SP_v1.4.6.3 +leak_corr,
  npatch=100) — confirm the canonical validation set.
- include_pseudo_cl defaults false (expensive NaMaster step, commented out in
  the live driver). Flip on when the C_l^BB column is wanted in the summary.
- End-to-end execution not run (needs container scientific stack); validated by
  dry-run only.
