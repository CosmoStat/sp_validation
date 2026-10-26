# Tests and production checks

## Round 4: independent data-table policy

All scientific work used allocation **1037411**, four cores on n09, the prescribed container/binds and `TMPDIR=/n17data/cdaley/tmp`.
Outputs and logs are `/n17data/cdaley/scratch/response-pr/round4/`; no scientific stack ran on the login node.
The self-built m/c values and all deleted estimates reproduce round 3 exactly on every grid (`baseline_checks.json`).
The Gaussian toy, independent/pooled tables and release-compatibility scan are described in [README.md](README.md#independent-training-and-self-weighting-round-4).

The final `python -m pytest -m 'not slow' --no-cov -q` returned **573 passed, 4 failed, 2 skipped, 7 deselected, 2 xfailed, 1 xpassed**, in **175.26 s** (`fast-suite-final.log`).
The four failures are the same path/schema and two temporary-directory cleanup failures independently reproduced on unchanged develop below; none was suppressed.
The focused suite passed **19 tests** (`focused-final.log`).
Four new regressions verify that external-table weights never train on simulation shapes; their shared-tile deletions match rebuilding each retained sample with the same external product; products without coordinate metadata are refused; the image-sim default requires a named product and preserves an explicit self-built diagnostic; and config staging replaces an old symlink without modifying its shared target.
The external-table test also exercises the catalogue driver's config-to-FITS-header path, including the named-product mapping, then saves and checks the fixed-external policy, product name, table and calibrated deleted means (`external-final.log`: **4 passed**).

The image-sim Snakemake diagnostic-config dry-run parses and schedules `im_init`; the default external config with no product fails at DAG parse with the intended error.
The container lacks Snakemake itself, so these two non-scientific DAG checks append the installed host Snakemake Python 3.12 site-packages to `sys.path`, without changing the image or scientific package stack.
`workflow-smoke.log` and `workflow-requires-product.log` record them; the latter's nonzero exit is expected.
`ruff check` and pre-commit pass on the changed Python files.

## Rounds 2–3

All round-2 and round-3 scientific execution used allocation **1037411** on n09, with `srun --jobid=1037411 -n 1 -c 4` inside `~/.cache/sp_validation/sp_validation.sif`.
`TMPDIR` was `/n17data/cdaley/tmp`; no scientific stack was imported on the login node.

The final `python -m pytest -m 'not slow' --no-cov -q` run returned **568 passed, 4 failed, 2 skipped, 7 deselected, 2 xfailed, 1 xpassed**, in **172.16 s**, with the chosen default N_min=3.
The log is `/n17data/cdaley/scratch/response-pr/round3/fast-suite-final.log`.
The five new sparse-table cases all pass: pooled moments bound the pathological fixture's W, repeated/deletion/reordered fits are deterministic, fully populated tables are bitwise unchanged, windows grow to the threshold (including a smaller-than-threshold total sample), and production refits match rebuilding the retained sample and save the pooling metadata.
The column-schema test recognizes `DESNMIN` and `DESMERGE` as FITS headers, not catalogue columns.
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
All four round-2 N_min=0 full-population m/c values, errors and covariance elements agree with the direct refit-W calculation to `atol=1e-13`.
Logs and results are `/n17data/cdaley/scratch/response-pr/round2/production_{1,2,3,4}.log` and `production_grid_{1,2,3,4}.json`.
Grid 4 also passes the same production FITS-to-m/c check with the chosen N_min=3; the fixture writes its newly calibrated e1/e2 and weights instead of retaining the old table's catalogue columns.
Its log is `/n17data/cdaley/scratch/response-pr/round3/production4.log`, with [saved production results](sparse_production_grid_4.json).
All m/c values, refit-W errors and covariance elements match the direct minimum-count measurement within 1e-13.
This catches the distinction between changing only R and changing both R and the numerator weights when W is refitted.

The earlier actual `scripts/calibration/calibrate_comprehensive_cat.py` driver smoke remains under `/n17data/cdaley/scratch/response-pr/driver-smoke/`.
It established catalogue-driver integration, string TILE_ID, corrected exported w_iv and the existing PSF-leakage path; it does not claim to fix separately tracked leakage defects.
The new all-grid FITS checks establish the primary refit-W uncertainty path.

`ruff check` and pre-commit pass on the changed Python files.
