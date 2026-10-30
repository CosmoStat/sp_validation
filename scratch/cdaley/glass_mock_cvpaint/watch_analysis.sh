#!/bin/bash
set -euo pipefail
ROOT=/n17data/cdaley/scratch/glass-mock-cv/cvpaint
export TMPDIR=/n17data/cdaley/scratch/glass-mock-cv/tmp
wait_for() { while [[ ! -s "$1" ]]; do sleep 30; done; sleep 5; }
for v in GLASS1_clean GLASS1_blinded; do
 wait_for "$ROOT/output/pseudo_cl_${v}_powspace_nbins=32.sacc"
 wait_for "$ROOT/output/pseudo_cl_cov_${v}_powspace_nbins=32.fits"
done
srun --jobid=1045571 -n 1 -c 1 "$ROOT/analyze.sh" harmonic > "$ROOT/logs/validate_harmonic.log" 2>&1
wait_for "$ROOT/output/pseudo_cl_GLASS1_addB_powspace_nbins=32.sacc"
wait_for "$ROOT/output/pseudo_cl_cov_GLASS1_addB_powspace_nbins=32.fits"
srun --jobid=1045571 -n 1 -c 1 "$ROOT/analyze.sh" additive > "$ROOT/logs/validate_additive.log" 2>&1
