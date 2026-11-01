#!/bin/bash
#SBATCH -J gm-drive
#SBATCH -p comp
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --exclude=n17,n36
#SBATCH -o /n17data/cdaley/scratch/glass-mock-cv/cvpaint/logs/dispatch_63f9b.log
# Drives the painted-mock suite on run/glassmock-int (integration 63f90fe8) into allocation 1047200 from a batch job,
# so no login-side process (srun client, nohup, tmux) is load-bearing.
set -uo pipefail
for v in $(env | grep -o "^SLURM_[A-Z_]*"); do unset "$v"; done
ROOT=/n17data/cdaley/scratch/glass-mock-cv/cvpaint
JOB=1047200
export TMPDIR=/n17data/cdaley/scratch/glass-mock-cv/tmp
echo "start $(date) on $(hostname)"
srun --jobid=$JOB -n 1 -c 46 --overlap "$ROOT/run.sh" --rerun-incomplete cosmo_val_all \
  > "$ROOT/logs/all_63f9b.log" 2>&1
rc=$?; echo "cosmo_val_all rc=$rc $(date)"
[ $rc -eq 0 ] && echo 63f90fe8 > "$ROOT/workflow_complete"
for part in psf xi decompositions scan; do
  srun --jobid=$JOB -n 1 -c 1 --overlap "$ROOT/analyze.sh" $part \
    > "$ROOT/logs/validate_${part}_63f9b.log" 2>&1
  echo "analyze $part rc=$? $(date)"
done
[ $rc -eq 0 ] && touch "$ROOT/validation_complete"
echo "done $(date)"
