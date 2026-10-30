#!/bin/bash
set -euo pipefail
ROOT=/n17data/cdaley/scratch/glass-mock-cv/pte
N=${1:-10}
python "$ROOT/prepare.py" --seeds "$N"
targets=()
for ((i=1; i<=N; i++)); do
  printf -v seed '%02d' "$i"
  targets+=("$ROOT/output/pseudo_cl_GLASSpte_s${seed}_powspace_nbins=32.sacc" "$ROOT/output/GLASSpte_s${seed}_xi_minsep=1.0_maxsep=250.0_nbins=20_npatch=1.sacc")
done
base=covariance_GLASSpte_s01_ng_minsep=1.0_maxsep=250.0_nbins=20_masked
targets+=("$ROOT/output/pseudo_cl_cov_GLASSpte_s01_powspace_nbins=32.fits" "$ROOT/inference/data/covariance/$base/$base.dat" "$ROOT/output/GLASSpte_s01_pairs_minsep=1.0_maxsep=250.0_nbins=1000.txt")
srun --jobid=1045571 -n 1 -c 20 "$ROOT/run.sh" --rerun-incomplete "${targets[@]}"
for script in audit_inputs.py validate.py; do
  srun --jobid=1045571 -n 1 -c 20 apptainer exec --cleanenv --bind /n17data,/n09data,/automnt,/home \
    /home/cdaley/.cache/sp_validation/sp_validation.sif env \
    PYTHONPATH=/n17data/cdaley/scratch/glass-mock-cv/pte-wt/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=20 \
    MPLCONFIGDIR=/n17data/cdaley/scratch/glass-mock-cv/.mpl TMPDIR=/n17data/cdaley/scratch/glass-mock-cv/tmp \
    python -u "$ROOT/$script" --seeds "$N"
done
