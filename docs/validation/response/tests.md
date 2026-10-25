# Tests and production-driver smoke

All scientific execution used allocation **1037389** on n09 with `srun --jobid=1037389 -n 1 -c 4` inside `~/.cache/sp_validation/sp_validation.sif`.
`TMPDIR` was `/n17data/cdaley/tmp`; no scientific stack was imported on the login node.

The focused response suite passes **26 tests** (17.29 s): the three imported regression files, analytic split/joint/full-matrix/c tests, pure cell migration and sheared-only rows, spatial covariance against brute-force deletions, coherent-group errors versus object deletions, all four downstream reader adapters, and pair tensors against brute force.
The original regression assertions remain; fixtures now supply compatible weight/population metadata and production-fitted DES weights reproduce the original specified 1:3 population weights.
No xfail remains in those three regression files.
Fixed-W errors are tested; W refitting is deliberately not implemented.

The final full `pytest -m 'not slow' -p no:cacheprovider --no-cov -q` run at `2f3dd142` returned **553 passed, 4 failed, 2 skipped, 7 deselected, 2 xfailed, 1 xpassed**, in **162.01 s**.
All four failures were independently reproduced on unchanged develop, as detailed below.
The xfail/xpass results are outside the three imported regressions.

## Failures also observed on unchanged develop

An independent archive of `origin/develop` at `058da2ef` reproduced all four of these failures under the same allocation/container/binds:

- `test_configured_paths_exist_on_candide`: unavailable configured data/chain/mask paths (including /n09data and collaborators' home paths).
- `test_cat_config_columns_exist_on_candide`: “known column gaps now pass; drop them” for `SP_v1.4.6_glass_mock.shear`.
- `test_rho_tau_loader_reads_hdf5_stars_by_config_names[dataset]`.
- `test_rho_tau_loader_reads_hdf5_stars_by_config_names[chunks]`.

The last two fail on `TemporaryDirectory.cleanup()` with `OSError: [Errno 39] Directory not empty` under `/n17data/cdaley/tmp`, after their reader assertions.
The failures are observed on the base branch too; they are not suppressed or counted as passing here.
Logs are `/n17data/cdaley/scratch/response-pr/baseline-failures.log`, `focused-final.log` and `fast-suite-head-final.log`.

## Production calibration driver

The actual `scripts/calibration/calibrate_comprehensive_cat.py` was also executed, using a scratch copy of grid 1's no-external-shear config with DES case d and `additive_correction: false`.
The comprehensive catalogue remained a symlink to Martin's original, not a copied catalogue.
The output has **98,608 selected rows**, corrected `w_iv`, string TILE_ID, **39 tile deletions**, and a **20×20 response/c covariance**.
Its `e1`, `e2`, `w_des` and corrected `w_iv` match the independent grid-validation driver to `rtol=1e-12, atol=1e-14`.
The existing PSF-leakage path runs unchanged; this test does not claim to fix the separately tracked leakage defects.
Outputs and logs are under `/n17data/cdaley/scratch/response-pr/driver-smoke/` and `driver-smoke*.log`.

`ruff check` passes for all changed Python files, and pre-commit passes on the commits.
