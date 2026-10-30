# Painted GLASS PTE calibration

This is a mock-only run on the v1.4.6.3 footprint.
The registry uses `blind: none` only for painted catalogues with known truth.
The checkout is `/n17data/cdaley/scratch/glass-mock-cv/pte-wt`, branch `pte/glass`, with `run/glassmock-int` commit `4893fab9` merged at `de57b450`.
Scientific jobs run on allocation 1045571, inside `/home/cdaley/.cache/sp_validation/sp_validation.sif`, and use at most 20 cores.
The host Snakemake driver uses the default profile to wrap scientific rules in the image; `/scratchc03` is omitted because it is absent on n19.

## Run

After all requested painted mocks exist, run the following one command from the login node, changing 10 to 100 for 100 seeds.
It constructs the isolated registry, runs the workflow targets on the allocation, checks input invariance and noise moments, and then produces metrics and figures.
This end-to-end continuation has not yet been validated; inspect the log and figures before interpreting the numbers.

```bash
/n17data/cdaley/scratch/glass-mock-cv/pte/production.sh 10 2>&1 | tee /n17data/cdaley/scratch/glass-mock-cv/pte/production.log
```

The current detached run was started with `nohup .../production.sh 10 > .../production.log 2>&1 < /dev/null &` as login PID 1104723, allocation step `1045571.104`.
Inspect `production.log` and `squeue -j 1045571 -s` for progress.
Do not launch a second copy while it is running.
If the detached run exits after the Snakemake targets without running the post-processing scripts, rerun `production.sh 10` after the allocation step exits; Snakemake reuses completed outputs.
The run is scoped to harmonic EE/EB/BB and 20-bin ξ±, plus a seed-1 harmonic covariance and a non-Gaussian masked OneCovariance matrix with the reporting-grid pair counts.
One mock slot serializes scientific rules to prevent oversubscription.
No fine integration-grid ξ, COSEBIs, pure E/B or PSF statistics are requested.

## Inference and caveats

`validate.py` compares each `realization_ee`, which is already pixelized, with EE via the SACC bandpower window and with ξ± via a direct, finite-band flat-sky Bessel sum.
The latter avoids the numerical failure observed for CCL's FFTLog and Bessel transforms on some realized spectra.
For ξ± it averages the transform using measured weighted pairs in 1000 logarithmic bins, with the signal evaluated at each fine-bin center, and compares this to uniform annular weighting.
It does not evaluate individual pair separations and uses a flat-sky approximation.
It also calculates the EE comparison against `cl_ee_pixel`, the GLASS shell-ensemble target.

The harmonic iNKA covariance includes fiducial cosmological signal as well as shape noise.
Its catalogue covariance workspace retains the painted weighted ellipticities for its noise terms, so that input is seed-dependent even for identical positions, weights and intrinsic ellipticity amplitudes.
`audit_inputs.py` hashes positions and weights and records the painted shape moment per seed; it does not establish exact seed-independence of the harmonic covariance.
Reuse of the seed-1 matrix is therefore an approximation whose adequacy must be quantified.
The OneCovariance inputs use the shared survey geometry, n(z), cosmology and fixed per-component dispersion.
OneCovariance includes cosmic variance, non-Gaussian terms and super-sample covariance in its selected `ng` configuration.

Subtracting a realization's full-sky spectrum and using an ensemble covariance does not automatically yield a uniform PTE: the covariance is not conditioned on that realization, and the footprint samples only part of that sky.
Interpret those PTEs as a diagnostic, not as a formally exact conditional calibration test.
With only ten skies, the KS test has little power and its p-value must not be used to claim precision calibration.
