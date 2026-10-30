#!/bin/bash
set -euo pipefail
ROOT=/n17data/cdaley/scratch/glass-mock-cv/cvpaint
REPO=/n17data/cdaley/scratch/glass-mock-cv/run-int
export TMPDIR=/n17data/cdaley/scratch/glass-mock-cv/tmp
export XDG_CACHE_HOME="$ROOT/cache"
export MPLCONFIGDIR=/n17data/cdaley/scratch/glass-mock-cv/.mpl
export PYTHONPATH="$REPO/src"
export COSMO_VAL="$ROOT/output"
export COSMO_INFERENCE="$ROOT/inference"
export PYTHONUNBUFFERED=1
export APPTAINERENV_MPLCONFIGDIR="$MPLCONFIGDIR"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_TMPDIR="$TMPDIR"
export APPTAINERENV_OMP_NUM_THREADS=27
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_MKL_NUM_THREADS=1
cd "$REPO/papers/cosmo_val"
# One scientific producer at a time keeps native threads within our share.
exec snakemake "$@" --profile "$REPO/workflow/profiles/default" -s Snakefile \
  -j 27 --resources mock_slot=1 --default-resources mock_slot=1 \
  --apptainer-args '--cleanenv --bind /n17data,/n09data,/automnt,/home' \
  --set-threads pseudo_cl=27 pseudo_cl_cov=27 \
  --configfile "$ROOT/config.yaml"
