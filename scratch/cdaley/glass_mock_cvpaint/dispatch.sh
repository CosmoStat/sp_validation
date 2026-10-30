#!/bin/bash
set -euo pipefail
ROOT=/n17data/cdaley/scratch/glass-mock-cv/cvpaint
export TMPDIR=/n17data/cdaley/scratch/glass-mock-cv/tmp
JOB=${CVPAINT_JOB:-1047200}
OUT="$ROOT/output"
COV="$ROOT/inference/data/covariance"
stem=covariance_GLASS1_clean_g_minsep=0.08_maxsep=300.0_nbins=1000_masked
# Two explicitly disjoint DAGs: the covariance consumes existing pair counts,
# while the PSF branch independently measures each entry's rho/tau. 8+36=44.
# Only this covariance invocation skips the lock; never run a second dispatcher.
srun --jobid="$JOB" -n 1 -c 8 "$ROOT/run.sh" --nolock --rerun-incomplete \
 "$COV/$stem/$stem.dat" > "$ROOT/logs/integration_cov_sol.log" 2>&1 &
cov_pid=$!
trap 'kill -TERM "$cov_pid" 2>/dev/null || true' EXIT
srun --jobid="$JOB" -n 1 -c 36 "$ROOT/run.sh" --rerun-incomplete cv_rho_tau_fits \
 > "$ROOT/logs/psf_sol.log" 2>&1
srun --jobid="$JOB" -n 1 -c 1 "$ROOT/analyze.sh" psf > "$ROOT/logs/validate_psf_sol.log" 2>&1
wait "$cov_pid"
trap - EXIT
srun --jobid="$JOB" -n 1 -c 46 "$ROOT/run.sh" --rerun-incomplete \
 "$OUT/GLASS1_clean_pure_eb.sacc" "$OUT/GLASS1_clean_cosebis.sacc" \
 "$OUT/GLASS1_blinded_pure_eb.sacc" "$OUT/GLASS1_blinded_cosebis.sacc" \
 "$OUT/GLASS1_addB_pure_eb.sacc" "$OUT/GLASS1_addB_cosebis.sacc" \
 > "$ROOT/logs/decompositions_sol.log" 2>&1
srun --jobid="$JOB" -n 1 -c 1 "$ROOT/analyze.sh" decompositions > "$ROOT/logs/validate_decompositions_sol.log" 2>&1
srun --jobid="$JOB" -n 1 -c 46 "$ROOT/run.sh" --rerun-incomplete cosmo_val_all \
 > "$ROOT/logs/all_sol.log" 2>&1
touch "$ROOT/workflow_complete"
srun --jobid="$JOB" -n 1 -c 1 "$ROOT/analyze.sh" xi decompositions psf scan \
 > "$ROOT/logs/validate_final_sol.log" 2>&1
touch "$ROOT/validation_complete"
