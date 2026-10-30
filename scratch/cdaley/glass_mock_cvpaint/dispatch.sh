#!/bin/bash
set -euo pipefail
ROOT=/n17data/cdaley/scratch/glass-mock-cv/cvpaint
export TMPDIR=/n17data/cdaley/scratch/glass-mock-cv/tmp
while [[ ! -f "$ROOT/config.yaml" ]]; do sleep 10; done
OUT="$ROOT/output"
# Land the independent clean/blinded harmonic pair before any long NN passes.
srun --jobid=1045571 -n 1 -c 28 "$ROOT/run.sh" --rerun-incomplete \
 "$OUT/pseudo_cl_GLASS1_clean_powspace_nbins=32.sacc" \
 "$OUT/pseudo_cl_cov_GLASS1_clean_powspace_nbins=32.fits" \
 "$OUT/pseudo_cl_GLASS1_blinded_powspace_nbins=32.sacc" \
 "$OUT/pseudo_cl_cov_GLASS1_blinded_powspace_nbins=32.fits" \
 > "$ROOT/logs/harmonic_pair.log" 2>&1
# Harmonic injection and reporting xi land before the slow fine-grid counts.
srun --jobid=1045571 -n 1 -c 27 "$ROOT/run.sh" --rerun-incomplete \
 "$OUT/pseudo_cl_GLASS1_addB_powspace_nbins=32.sacc" \
 "$OUT/pseudo_cl_cov_GLASS1_addB_powspace_nbins=32.fits" \
 "$OUT/GLASS1_clean_xi_minsep=1.0_maxsep=250.0_nbins=20_npatch=1.sacc" \
 "$OUT/GLASS1_blinded_xi_minsep=1.0_maxsep=250.0_nbins=20_npatch=1.sacc" \
 > "$ROOT/logs/reporting_pair.log" 2>&1
# Land the reporting covariance and exact xi blind scatter check first.
COV="$ROOT/inference/data/covariance"
for v in GLASS1_clean GLASS1_blinded; do
 if [[ "$v" == GLASS1_blinded ]]; then
  while [[ ! -f "$ROOT/geometry_verified" ]]; do sleep 30; done
 fi
 stem="covariance_${v}_ng_minsep=1.0_maxsep=250.0_nbins=20_masked"
 srun --jobid=1045571 -n 1 -c 27 "$ROOT/run.sh" --rerun-incomplete "$COV/$stem/$stem.dat" \
  > "$ROOT/logs/reporting_cov_${v}.log" 2>&1
done
srun --jobid=1045571 -n 1 -c 1 "$ROOT/analyze.sh" blind_xi > "$ROOT/logs/validate_blind_xi.log" 2>&1
# Clean geometry and decomposition follow the full-geometry audit.
srun --jobid=1045571 -n 1 -c 27 "$ROOT/run.sh" --rerun-incomplete \
 "$OUT/GLASS1_clean_pure_eb.sacc" "$OUT/GLASS1_clean_cosebis.sacc" \
 > "$ROOT/logs/decomposition_clean.log" 2>&1
# Verified geometry-only references avoid repeated identical NN traversals.
while [[ ! -f "$ROOT/geometry_verified" ]]; do sleep 30; done
srun --jobid=1045571 -n 1 -c 27 "$ROOT/run.sh" --rerun-incomplete \
 "$OUT/GLASS1_blinded_pure_eb.sacc" "$OUT/GLASS1_blinded_cosebis.sacc" \
 > "$ROOT/logs/decomposition_blinded.log" 2>&1
srun --jobid=1045571 -n 1 -c 27 "$ROOT/run.sh" --rerun-incomplete cosmo_val_all \
 > "$ROOT/logs/all.log" 2>&1
touch "$ROOT/workflow_complete"
srun --jobid=1045571 -n 1 -c 1 "$ROOT/analyze.sh" harmonic additive xi decompositions psf scan \
 > "$ROOT/logs/validate_final.log" 2>&1
touch "$ROOT/validation_complete"
