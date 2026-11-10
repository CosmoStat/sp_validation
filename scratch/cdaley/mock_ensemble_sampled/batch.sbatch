#!/bin/bash
#SBATCH --account=rrg-kyi_cpu
#SBATCH --job-name=mock-sampled
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=700G
#SBATCH --time=12:00:00
#SBATCH --output=logs/batch-%j.log
# From the run root: sbatch --export=ALL,ROOT=$PWD,START=2,END=5 batch.sbatch
set -euo pipefail
: "${ROOT:?}" "${START:?}" "${END:?}"
export MEM_MB=680000
bash /project/rrg-kyi/cdaley/unions/code/wt-mockens-cv/scratch/cdaley/mock_ensemble_sampled/run.sh
