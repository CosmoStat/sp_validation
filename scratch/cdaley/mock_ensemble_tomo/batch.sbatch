#!/bin/bash
#SBATCH --account=def-sfabbro
#SBATCH --job-name=mock-tomo-batch
#SBATCH --cpus-per-task=192
#SBATCH --mem=700G
#SBATCH --time=24:00:00
#SBATCH --output=/project/6001537/cdaley/mock-ensemble-tomo/logs/batch-%j.log
set -euo pipefail
HERE=/project/6001537/cdaley/sp/sp_validation-mockens/scratch/cdaley/mock_ensemble_tomo
: "${START:?Supply START and END with sbatch --export=ALL,START=2,END=20}"
: "${END:?Supply START and END}"
export MEM_MB=680000
/home/cdaley/sp/snakemake-env/bin/python -c "import json; assert json.load(open('/project/6001537/cdaley/mock-ensemble-tomo/seed1_known_answer.json'))['passed']"
bash "$HERE/run.sh"
