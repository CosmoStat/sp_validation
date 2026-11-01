#!/bin/bash
# Host Snakemake on the default profile; every job runs inside the image.
set -euo pipefail
source "$(dirname "$0")/env.sh"
export COSMO_VAL="$CVPAINT_ROOT/output" COSMO_INFERENCE="$CVPAINT_ROOT/inference" PYTHONPATH="$REPO/src"
export APPTAINERENV_MPLCONFIGDIR="$MPLCONFIGDIR" APPTAINERENV_TMPDIR="$TMPDIR" APPTAINERENV_PYTHONUNBUFFERED=1
THREADS=${CVPAINT_THREADS:-${SLURM_CPUS_PER_TASK:-8}}
export APPTAINERENV_OMP_NUM_THREADS="$THREADS" APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_MKL_NUM_THREADS=1
cd "$REPO/papers/cosmo_val"
# One scientific producer at a time keeps native threads within the allocation.
exec "$SNAKEMAKE" "$@" --profile "$REPO/workflow/profiles/default" -s Snakefile \
  -j "$THREADS" --resources mock_slot=1 --default-resources mock_slot=1 \
  --apptainer-args "--cleanenv --bind $BINDS" \
  --set-threads pseudo_cl="$THREADS" pseudo_cl_cov="$THREADS" \
  --configfile "$CVPAINT_ROOT/config.yaml"
