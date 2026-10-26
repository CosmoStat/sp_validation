# Tests and production checks

All round-2 scientific execution used allocation **1037411** on n09, with `srun --jobid=1037411 -n 1 -c 4` inside `~/.cache/sp_validation/sp_validation.sif`.
`TMPDIR` was `/n17data/cdaley/tmp`; no scientific stack was imported on the login node.

The final `python -m pytest -m 'not slow' --no-cov -q` run returned **563 passed, 4 failed, 2 skipped, 7 deselected, 2 xfailed, 1 xpassed**, in **174.94 s**.
The log is `/n17data/cdaley/scratch/response-pr/round2/fast-suite-final.log`.
The four failures match the independently reproduced develop failures below; none was suppressed or counted as passing.
The xfail/xpass results are outside the three imported response regressions.

The seven additional passing cases cover branch-specific squared IV errors and normalized selection response, refusal of missing IV branch errors, carrying available upstream errors without blocking historical DES inputs, refit-W deletion versus calibration of the retained raw sample, constant/varying scalar pair responses and every shared-patch deletion versus explicit pair sums, and production m/c consumption of refitted numerator weights.
The original regression assertions remain; their fixtures now include the branch errors required by exact IV.
No xfail remains in the three imported response regression files.
TreeCorr's float32 scalar-field precision is allowed at 1e-7 in the explicit-pair comparisons.

## Failures also observed on unchanged develop

An independent archive of `origin/develop` at `058da2ef` reproduced all four failures under the same container/binds during the preceding validation:

- `test_configured_paths_exist_on_candide`: unavailable configured data/chain/mask paths, including /n09data and collaborators' home paths.
- `test_cat_config_columns_exist_on_candide`: “known column gaps now pass; drop them” for `SP_v1.4.6_glass_mock.shear`.
- `test_rho_tau_loader_reads_hdf5_stars_by_config_names[dataset]`.
- `test_rho_tau_loader_reads_hdf5_stars_by_config_names[chunks]`.

The last two fail on `TemporaryDirectory.cleanup()` with `OSError: [Errno 39] Directory not empty` under `/n17data/cdaley/tmp`, after their reader assertions.
Baseline logs are `/n17data/cdaley/scratch/response-pr/baseline-failures.log`, `focused-final.log` and `fast-suite-head-final.log`.

## Production response/weight uncertainty

The four-grid round trip uses the committed `save_response_product` and `ImageSimMBias`, not a separate error estimator.
Each grid's five external branches save refit-W `R_JK` responses and calibrated deleted means in FITS, then the normal production m reader computes the shared-tile (m1,c1,m2,c2) covariance.
All full-population m/c values, errors and covariance elements agree with the direct refit-W calculation to `atol=1e-13`.
Logs and results are `/n17data/cdaley/scratch/response-pr/round2/production_{1,2,3,4}.log` and `production_grid_{1,2,3,4}.json`.
This catches the distinction between changing only R and changing both R and the numerator weights when W is refitted.

The earlier actual `scripts/calibration/calibrate_comprehensive_cat.py` driver smoke remains under `/n17data/cdaley/scratch/response-pr/driver-smoke/`.
It established catalogue-driver integration, string TILE_ID, corrected exported w_iv and the existing PSF-leakage path; it does not claim to fix separately tracked leakage defects.
The new all-grid FITS checks establish the primary refit-W uncertainty path.

`ruff check` and pre-commit pass on the changed Python files.
