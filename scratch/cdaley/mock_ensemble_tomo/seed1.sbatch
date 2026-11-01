#!/bin/bash
#SBATCH --account=def-sfabbro
#SBATCH --job-name=mock-tomo-seed1
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=08:00:00
#SBATCH --output=/project/6001537/cdaley/mock-ensemble-tomo/logs/seed1-%j.log
set -euo pipefail
HERE=/project/6001537/cdaley/sp/sp_validation-mockens/scratch/cdaley/mock_ensemble_tomo
export START=1 END=1 MEM_MB=280000
REPO=/project/6001537/cdaley/sp/sp_validation-mockens
APPTAINER=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/apptainer/1.3.4/bin/apptainer
SIF=/project/6001537/cdaley/containers/sp_validation-ddd58fe0.sif
$APPTAINER exec --cleanenv --bind /project,/home "$SIF" env PYTHONPATH=$REPO/src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest "$REPO/src/sp_validation/tests/test_cosmo_val.py" -k 'shear_measurement_does_not_read_stars or published_means_are_layout_independent' -q --no-cov
bash "$HERE/run.sh"
$APPTAINER exec --cleanenv --bind /project,/home "$SIF" env PYTHONPATH=$REPO/src OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python "$HERE/known_answer.py"
