#!/bin/zsh
#SBATCH --output=/home/guerrini/compute_theory_cov.out
#SBATCH --error=/home/guerrini/compute_theory_cov_err.out
#SBATCH --partition=comp,pscomp
#SBATCH --job-name=compute_theory_cov
#SBATCH --ntasks-per-node=48
#SBATCH --time=48:00:00


module purge
module load intelpython3

source activate sp_validation

cd ~/sp_validation/notebooks/cosmo_val

python compute_theory_cov.py

#cosmosis cosmosis_config/cosmosis_pipeline_SP_v1.3_LFmask_8k.ini


exit 0