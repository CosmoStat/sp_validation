#!/bin/bash
#SBATCH --job-name=unions_V1.4
#SBATCH --mail-user=lgoh@roe.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=compl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=4-00:00:00 
#SBATCH --output=/n23data1/n06data/lgoh/scratch/CFIS-UNIONS/chains/SP_v1.4_A/inference_A.log

module load gcc
module load intelpython/3-2024.1.0
module load openmpi
source cosmosis-configure
source activate my_env

cosmosis --mpi /n23data1/n06data/lgoh/scratch/CFIS-UNIONS/CFIS-UNIONS_dev/cosmo_inference/cosmosis_config/cosmosis_pipeline_A_1.ini

# 
# Return exit code
exit 0