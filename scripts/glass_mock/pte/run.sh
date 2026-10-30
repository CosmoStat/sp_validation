#!/bin/bash
set -euo pipefail
ROOT=/n17data/cdaley/scratch/glass-mock-cv/pte
REPO=/n17data/cdaley/scratch/glass-mock-cv/pte-wt
export TMPDIR=/n17data/cdaley/scratch/glass-mock-cv/tmp
export XDG_CACHE_HOME="$ROOT/cache" MPLCONFIGDIR=/n17data/cdaley/scratch/glass-mock-cv/.mpl PYTHONPATH="$REPO/src"
export COSMO_VAL="$ROOT/output" COSMO_INFERENCE="$ROOT/inference" PYTHONUNBUFFERED=1
export APPTAINERENV_MPLCONFIGDIR="$MPLCONFIGDIR" APPTAINERENV_PYTHONUNBUFFERED=1 APPTAINERENV_TMPDIR="$TMPDIR"
export APPTAINERENV_OMP_NUM_THREADS=20 APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_MKL_NUM_THREADS=1
cd "$REPO/papers/cosmo_val"
exec snakemake "$@" --profile "$REPO/workflow/profiles/default" -s Snakefile -j 20 \
  --resources mock_slot=1 --default-resources mock_slot=1 \
  --apptainer-args '--cleanenv --bind /n17data,/n09data,/automnt,/home' \
  --set-threads pseudo_cl=20 pseudo_cl_cov=20 xi=20 covariance_pair_counts=20 \
  --configfile "$ROOT/config.yaml"
