# Nibi bound-9 tomographic mock ensemble

This recipe mirrors `/project/6001537/cdaley/cosmo-val-v2-tomo` without writing into it.
The data worker archived the brief's completed run under `checks/before-develop-20ba864a` before rerunning develop.
The registry, configuration and covariance references are pinned to that snapshot, so concurrent data work cannot replace the ensemble's reference covariance.
All outputs and the throwaway `mock-ensemble-tomo` blind live under `/project/6001537/cdaley/mock-ensemble-tomo`.
The input shapes enter only `paint_catalogue`, which rotates them before composing the mock.
The source catalogue's shapes and the `unions` blind are never analysed or opened.

## Launch

Run `prepare.py --seeds 100` in the analysis container on a compute allocation.
It copies the data run's entry, n(z), masks, cosmology and analytic covariance references into the mock registry.
It creates the mock-only blind through `sp_validation.blinding.init` if it doesn't exist.

Submit `sbatch seed1.sbatch` from this directory.
The driver runs local Snakemake inside one allocation, with explicit CPU and memory resources.
Seed 1 includes an independently measured clean twin and a full geometry hash against source rows in bins 1–4.
Run `known_answer.py` in the container on an allocation after the seed-1 targets complete.
It writes `RESULT.md`, the per-pair table in `seed1_known_answer.json`, and three known-answer figures.
Do not scale before that gate passes.

Then run `sbatch --export=ALL,START=2,END=20 batch.sbatch`.
Raise END as needed to grow toward 100.
Resubmitting resumes completed Snakemake products; no job sleeps or polls internally.
If using the fallback account, pass `--account=def-mjhudson` to sbatch.

## Products and cost

- `seeds/seed_{s:05d}/maps/`: complex128 convergence alms, realised/theory auto and cross spectra, solver audit and generation times.
- `cosmo_val/MOCK_{s:05d}*`: per-statistic and final tomographic SACC, every per-pair decomposition NPZ (all configured COSEBIs cuts), and joint/per-pair B-null summary.
- `benchmarks/`: per-rule wall time, peak RSS and CPU measurements.
- `geometry_audit.json`: full retained-row position/weight/bin hashes, with no access to input shape columns.

Painted HDF5 catalogues are Snakemake temporary outputs and disappear only after their consumers complete.
Every seed regenerates locally from sky seed s and rotation seed 1000000+s.
No float32 sky maps are made.
All mock entries read the data's existing analytic tomographic covariances; the dry-run contains no covariance rule.

Reporting ξ uses `npatch=1` instead of 100 to omit a discarded jackknife pass.
`process_gg` publishes unpatched means regardless of npatch, so this changes no reported correlation, pair weight or separation.
The final SACC receives the data's analytic covariance, not the part's cheap shot-noise placeholder.
The reporting and integration angular grids, approximation controls and final analytic covariance remain the data run's.
No PSF, leakage, rho/tau or workflow plotting targets run.

## Known-answer convention

The prediction is the seed's own full-sky realised EE auto/cross-spectrum, not the ensemble mean.
Pseudo-Cℓ uses the estimator's persisted bandpower windows.
ξ± uses a finite direct J0/J4 sum over ell=2..8192 with ell+0.5, averaged using the actual fine-grid pair weights and fractional annular overlap.
It uses neither FFTLog nor an extra pixel window.
Footprint cosmic variance remains, so these comparisons are consistency checks under the data's covariance.
The gate rejects catastrophic E-recovery PTEs below 1e-4, not low B-null PTEs: their frequency is the ensemble's scientific question.
The independently measured clean twin must reproduce the mock blind's designed xi/EE shift and unchanged B to numerical precision.
