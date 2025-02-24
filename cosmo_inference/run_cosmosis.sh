#!/bin/zsh
#SBATCH --output=/n17data/sguerrini/cosmosis_fake_sp_v1.4.1_scale_cut_2.out
#SBATCH --error=/n17data/sguerrini/cosmosis_fake_sp_v1.4.1_scale_cut_2.out
#SBATCH --partition=comp,pscomp
#SBATCH --job-name=cosmosis_fake_sp_v1.4.1_scale_cut_2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

module purge
module load intelpython

module load openmpi

source activate sp_validation

source cosmosis-configure

cd ~/sp_validation/cosmo_inference

# Just 1 for OMP_NUM_THREADS for this Python script
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# And let the low-level threading use all of the requested cores
export OPENBLAS_NUM_THREADS=$NSLOTS

mpirun --np 4 cosmosis --mpi cosmosis_config/cosmosis_pipeline_fake_SP_v1.4.1_2.ini

#cosmosis cosmosis_config/cosmosis_pipeline_SP_v1.3_LFmask_8k.ini


exit 0
