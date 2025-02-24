#!/bin/zsh
#SBATCH --output=/home/guerrini/cosmo_val_run_th.out
#SBATCH --error=/home/guerrini/cosmo_val_run_th.err
#SBATCH --partition=comp,pscomp
#SBATCH --job-name=cosmo_val_run_th
#SBATCH --tasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=48:00:00


module purge
module load intelpython3

source activate sp_validation

cd ~/sp_validation/notebooks/cosmo_val

python cosmo_val.py

#cosmosis cosmosis_config/cosmosis_pipeline_SP_v1.3_LFmask_8k.ini


exit 0