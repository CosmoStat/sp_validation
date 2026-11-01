#!/bin/bash
#SBATCH --account=def-sfabbro
#SBATCH --job-name=mock-tomo-seed1
#SBATCH --cpus-per-task=72
#SBATCH --mem=360G
#SBATCH --time=12:00:00
#SBATCH --output=/project/6001537/cdaley/mock-ensemble-tomo/logs/seed1-%j.log
set -euo pipefail
HERE=/project/6001537/cdaley/sp/sp_validation-mockens/scratch/cdaley/mock_ensemble_tomo
export START=1 END=1 MEM_MB=350000
bash "$HERE/run.sh"
