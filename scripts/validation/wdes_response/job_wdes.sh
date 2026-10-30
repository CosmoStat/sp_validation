#!/usr/bin/bash
#SBATCH --output=/n23data1/mkilbing/imsim_wdes_response/job_%x_%A_%a.out
#SBATCH --partition=comp,pscomp
#SBATCH --job-name=wdes_resp
#SBATCH --array=1-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --exclude=n09,n17,n23,n24,n25,n34,n35,n36
set -euo pipefail
export SCR="/n23data1/mkilbing/shapepipe/tmp/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$SCR"
trap 'rm -rf "$SCR"' EXIT
export TMPDIR="$SCR" APPTAINER_TMPDIR="$SCR" SINGULARITY_TMPDIR="$SCR"

G=$SLURM_ARRAY_TASK_ID
W=/n23data1/mkilbing/imsim_wdes_response
OUT=$W/grid_$G
SPV=/n17data/mkilbing/astro/repositories/github/sp_validation_spv
GRID=/n17data/mkilbing/astro/Runs/shapepipe/CFIS/v2.0/image_sims/new_workflow/grid_$G
SIF=/n17data/cdaley/containers/sp_validation_im_sims.sif
mkdir -p $OUT

run() {
  apptainer exec --cleanenv --bind /n17data,/n23data1,/n09data,/home \
    --env TMPDIR=$SCR,MPLCONFIGDIR=$SCR,XDG_CACHE_HOME=$SCR,PYTHONPATH=$SPV/src,OMP_NUM_THREADS=1,HOME=$HOME,USER=$USER \
    $SIF "$@"
}

run python $W/wdes_response.py $GRID $OUT
for S in iv iv_d iv_box des_c des_d; do
  run python $W/mbias_config.py $G $S $OUT
  cd $OUT/$S
  run python $SPV/scripts/compute_m_bias_image_sims.py -c $OUT/$S/m_bias_config.yaml
done
run python $W/diff_jk.py $OUT iv iv_d iv iv_box des_c des_d
echo "exit code 0"
