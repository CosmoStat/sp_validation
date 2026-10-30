#!/bin/bash
set -euo pipefail
ROOT=/n17data/cdaley/scratch/glass-mock-cv/cvpaint
export TMPDIR=/n17data/cdaley/scratch/glass-mock-cv/tmp
exec apptainer exec --bind /n17data,/n09data,/automnt,/home \
 ~/.cache/sp_validation/sp_validation.sif env \
 PYTHONPATH=/n17data/cdaley/scratch/glass-mock-cv/run-int/src \
 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 MPLCONFIGDIR=/n17data/cdaley/scratch/glass-mock-cv/.mpl TMPDIR="$TMPDIR" \
 python -u "$ROOT/validate.py" "$@"
