# Painted seed-1 mocks through cosmo_val

This isolated run tests four known-truth mocks, not real-catalogue cosmological vectors.
Real PSF/star catalogues are read in place for the PSF tests.
The input maps explicitly use the **v1.4.6.3 B n(z)** (`maps/maps.json:config.nz_path`), gamma rather than reduced shear, nside 4096 and lmax 8192.
The realization-level answers are pixelized and take no additional generation pixel window.
The catalogue harmonic estimator takes no measurement-map pixel window.

## Inputs

- Mock directory: `/n17data/cdaley/scratch/glass-mock-cv/mocks/v1.4.6.3/seed_00001/`.
- Entries: `GLASS1_clean` (`painted.h5`), `GLASS1_blinded` (`catalogues/painted_blinded.h5`, physical byte copy with a distinct realpath), `GLASS1_psfleak` (`painted_psfleak.h5`), `GLASS1_addB` (`painted_addB.h5`).
- Shears `e1/e2`, weights `w_des`, response R=1 and galaxy-position PSF `e1_PSF/e2_PSF`.
- The `cat_config.yaml` PSF/star and covariance survey metadata are copied from the repository's `SP_v1.4.6.3` entry.
The paint report gives exactly the same isotropic noise dispersion; this run does not estimate a real-data E vector.
- `paths.blinds` points only at `/n17data/cdaley/scratch/glass-mock-cv/cvmock/blinds`, the predecessor's mock-only registry.
Its existing `glassmock-test` record is reused, never printed or committed.
- Independent producers measure rho/tau for all four entries.
No rho/tau cache is staged between entries.

## Checkout and image

Checkout: `/n17data/cdaley/scratch/glass-mock-cv/run-int`, branch `run/glassmock-int`.
Starting point `4662cbbe`; painted-generator/reader merged from `a6741eef`; integration merged from `29f962bb` (including develop's no-leak gate).
The run recipe and audit sources are versioned under `scratch/cdaley/glass_mock_cvpaint`, initially at `85d3e2bd`.
The inherited checkpoint is `a7b1cbfe`; `ee8ce619` adds the integration-covariance reference and its ten focused tests.
The continuation commits update the dispatch/thread settings and the finite-Bessel ξ validator.
`93ffebb7` adds a conservative numerical-scale prefilter to the existing leak oracle and filters nonzero finite mock references; `a7b1cbfe` retains the complete existing decoder coverage alongside new boundary/performance tests.
All 50 scanner/producer-gate tests pass in the image (`logs/blind_scan_tests_complete.log`).
The prefilter cannot discard an `isclose` match: it uses outward-rounded bounds on the possible absolute values, then applies the same comparison and decoder tolerances.
Merge commit `1e465cb6` retains the isolated-registry/rerun fixes.
Commit `4893fab9` keeps the integration harmonic estimator, covariance and SACC-writer interfaces consistent after resolving the catalogue-only branch divergence.
The integration's retained catalogue fields and zero signal-only noise input fix are used for catalogue NaMaster covariance.

Image: `/home/cdaley/.cache/sp_validation/sp_validation.sif`.
The inherited label receipt is `../cvmock/image.json` (image revision `4954b97268d0`); `image.json` here records the image inspected for this run.
`PYTHONPATH` selects this checkout's `src/`, not the image's installed package.
Commit `3ace2f14` adds opt-in geometry-only pair-count references, with three passing host tests and an actual paired-target dry run proving that only one pair-count producer is scheduled.
The image lacks Snakemake, so host Snakemake drives the default profile and runs each scientific job inside the SIF.
This is the inherited, documented host-driver/container-job arrangement.
No candide profile or new batch job is used.
`/scratchc03` is absent on n19; the other prescribed binds are retained.

## Commands

```bash
ROOT=/n17data/cdaley/scratch/glass-mock-cv/cvpaint
REPO=/n17data/cdaley/scratch/glass-mock-cv/run-int
export TMPDIR=/n17data/cdaley/scratch/glass-mock-cv/tmp
srun --jobid=1045571 -n 1 -c 1 \
  apptainer exec --bind /n17data,/n09data,/automnt,/home \
  ~/.cache/sp_validation/sp_validation.sif \
  env PYTHONPATH="$REPO/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/n17data/cdaley/scratch/glass-mock-cv/.mpl TMPDIR="$TMPDIR" \
  python "$ROOT/prepare.py"

# Verify all RA/Dec/weight values; only then enable geometry-only references.
srun --jobid=1045571 -n 1 -c 1 \
  apptainer exec --bind /n17data,/n09data,/automnt,/home \
  ~/.cache/sp_validation/sp_validation.sif \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR="$TMPDIR" \
  python -u "$ROOT/audit_geometry.py"

# Script records the exact targets and order; no scientific process runs here.
nohup "$ROOT/dispatch.sh" > "$ROOT/logs/dispatch.log" 2>&1 &

# Individual diagnostics are restartable once their inputs exist.
srun --jobid=1045571 -n 1 -c 1 "$ROOT/analyze.sh" harmonic
srun --jobid=1045571 -n 1 -c 1 "$ROOT/analyze.sh" additive
srun --jobid=1045571 -n 1 -c 1 "$ROOT/analyze.sh" xi decompositions psf scan
```

The initial stages used allocation 1045571 with at most 28 cores; that allocation and its old dispatcher were cancelled on 30 October.
The continuation uses allocation **1047200** on n30 (48 cores, 400 GB, two days).
`run.sh` takes `CVPAINT_THREADS`, falling back to `SLURM_CPUS_PER_TASK` then 27, and rejects values outside 1–46.
A one-slot Snakemake resource serializes scientific producers within each invocation.
The continuation runs two disjoint DAGs concurrently: rho/tau fits on 36 cores and the clean Gaussian integration covariance on 8 cores.
Only that covariance invocation uses `--nolock`, because its only unfinished output is disjoint from the PSF DAG.
The dispatcher validates PSF fits, waits for covariance, computes clean/blinded/addB decompositions on 46 cores, validates decompositions, completes `cosmo_val_all`, then runs final validation including the leak scan.
Use `export TMPDIR=/n17data/cdaley/scratch/glass-mock-cv/tmp` before `srun`; the login default isn't writable on n30.
The commands above record the original run; replace `--jobid=1045571` with `--jobid=1047200` for any continuation.
Do not rerun preparation against the active config: it would remove the explicit sharing settings.
Launch the continuation with `nohup "$ROOT/dispatch.sh" > "$ROOT/logs/dispatch_sol.log" 2>&1 &` only if no dispatcher is already running.
Weighted fine-grid pair counts require two traversals.
Once `audit_geometry.py` verifies every RA/Dec/weight value, `geometry_audit.json` records the full hashes and the config explicitly references the clean entry's geometry-only count table for the other entries.
No table is copied into another producer's outputs, and no shear or rho/tau value is shared.
Each entry retains its own shear, harmonic covariance and rho/tau producers.
`covariance.integration_reference` additionally makes the other entries consume the clean Gaussian integration covariance by direct input reference, without copying files.
This applies only to decomposition covariances; reporting non-Gaussian covariance producers remain independent.
The resolver requires identical `cov_th`, mask, n(z) path and an explicitly shared pair-count reference.
Ten focused host tests pass, including a dry-run DAG test proving that the requesting catalogue retains its own measured ξ (`logs/reference_tests_sol.log`).
The actual three-entry dry run schedules one covariance solve (`logs/reference_dryrun_sol.log`).
The suggested caches under `d2t-parity` and `e2e-253-v2/inf-a` aren't reused: their `.ini` files specify A n(z), whereas the mock requires B n(z).
The shape-noise covariance uses the workflow's measured weighted pair counts.

## Object-wise uncertainty audit

All audit scripts read mock shapes, mock truth or geometry/PSF columns only.
Run each with the same one-core image environment as the preparation command:

```bash
for script in early_objectwise.py objectwise_audit.py objectwise_patch_scales.py \
              objectwise_sky_ensemble.py objectwise_noise_free_formal.py; do
  srun --jobid=1045571 -n 1 -c 1 \
    apptainer exec --bind /n17data,/n09data,/automnt,/home \
    ~/.cache/sp_validation/sp_validation.sif \
    env PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 MPLCONFIGDIR=/n17data/cdaley/scratch/glass-mock-cv/.mpl \
    TMPDIR="$TMPDIR" python -u "$ROOT/$script"
done
```

`early_objectwise.py` calls the pipeline's actual mixed affine regression for
clean and PSF-injected entries; its fit caches are reusable by the final suite.
`objectwise_audit.py` independently accumulates weighted normal equations for
full mock, rotated noise and noise-free `g_true`, and stores per-cell moments.
`objectwise_patch_scales.py` uses those moments for nside 2/4/8 sensitivity checks.
The main spatial scales are nside 4 and 8; the few unequal nside-2 groups aren't
a default error choice.
`objectwise_sky_ensemble.py` builds an exact per-generation-pixel weighted PSF
operator and applies it to all ten saved gamma maps without rereading ten full
catalogues; its seed-1 check matches the stored-truth regression at roundoff.
`objectwise_noise_free_formal.py` calculates the independent-object Hessian error
for the same six-parameter model, using the pooled weighted residual variance
and 2N−6 degrees of freedom, and contrasts it with spatial errors.
`corr2d_source.txt` and `loss2d_source.txt` record the external fitter's model and
concatenated weighted residuals used to verify that calculation.

The scripts do not change the pipeline's default object-wise uncertainty.
The ten-sky variance and irregular-footprint jackknives are diagnostics, not an
ensemble-calibrated real-data uncertainty prescription.
See `RESULT.md` for the measured underestimation and its limits.

## Validation

`validate.py` writes `metrics.json` and known-answer figures.
The rejection criterion is a predeclared individual PTE <0.01, with primary EE cuts ell 300–1600 and theta 12–83 arcmin.
No cut is tuned after inspecting a result.
EE compares to both the sky's realized spectrum and its GLASS shell-ensemble target, passed through saved bandpower windows or pair-weighted xi transforms.
Full-sky realization subtraction does not condition the footprint covariance on the sky, so these PTEs are diagnostics rather than guaranteed uniform variates.
The ξ transform is a finite direct J0/J4 Bessel sum over ℓ=2–8192 using ℓ+½, with no FFTLog extrapolation of the jagged realized spectrum.
It is averaged over the measured 1000-bin integration-grid pair weights, using fractional annular overlap with reporting bins.
`logs/validate_xi_sol.log` records the successful run on 1047200; the resulting recovery and updated scatter figures were inspected.

The additive-B amplitude fit uses the clean single-catalogue BB covariance as the pipeline detection uncertainty.
It is not claimed to estimate the covariance of an identical-noise difference.
The difference includes injection×noise and injection×baseline-B cross terms.
The injected catalogue's own BB null PTE is also reported.

The blinded-tree scanner calls develop's `assert_no_raw_values` against clean reporting/integration xi, EE, COSEBIs E and pure-E references, with concealed twins supplied at decoder precision.
Because all four entries share an output root, `blinded_tree/` is a symlink view of every blinded-entry and shared product, excluding other entries' products.
`blind_scan_manifest.json` records the exact scope.
PNG metadata is decoded; image pixels are not an invertible numerical leak oracle.

Merge checks passed inside the image on allocation 1045571: five constructor/optional-PSF tests and 44 harmonic/SACC-writer/blind-leak tests.
See `logs/merge_tests.log` and `logs/harmonic_gate_tests.log`.
