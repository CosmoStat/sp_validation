#!/bin/bash
#SBATCH --account=rrg-kyi_cpu
#SBATCH --job-name=mock-sampled
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=700G
#SBATCH --time=12:00:00
#SBATCH --output=/project/rrg-kyi/cdaley/unions/analyses/mock-ensemble-sampled/logs/batch-%j.log
# sbatch --export=ALL,START=2,END=5 batch.sbatch
set -euo pipefail
: "${START:?}" "${END:?}"
export MEM_MB=680000
bash /project/rrg-kyi/cdaley/unions/code/wt-mockens-cv/scratch/cdaley/mock_ensemble_sampled/run.sh
