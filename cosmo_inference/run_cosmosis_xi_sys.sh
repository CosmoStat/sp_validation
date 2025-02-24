#!/bin/zsh
#SBATCH --output=/n17data/sguerrini/cosmosis_psfex_xi_sys_cut_80.out
#SBATCH --error=/n17data/sguerrini/cosmosis_psfex_xi_sys_cut_80.out
#SBATCH --partition=comp,pscomp
#SBATCH --job-name=cosmosis_psfex_xi_sys_cut_80
#SBATCH --ntasks-per-node=20
#SBATCH --mem=64G
#SBATCH --time=48:00:00

module purge
module load intelpython

module load openmpi

source activate sp_validation

source cosmosis-configure

cd ~/sp_validation/cosmo_inference

# Just 1 for OMP_NUM_THREADS for this Python script
export OMP_NUM_THREADS=1
# And let the low-level threading use all of the requested cores
export OPENBLAS_NUM_THREADS=$NSLOTS

mpirun --np 20 cosmosis --mpi cosmosis_config/cosmosis_pipeline_SP_v1.4-P1+3_xi_sys.ini

#cosmosis cosmosis_config/cosmosis_pipeline_SP_v1.3_LFmask_8k.ini


exit 0
