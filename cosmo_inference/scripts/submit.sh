#!/bin/bash
#SBATCH --job-name=shapepipe_matching
#SBATCH --mail-user=lisa.goh@cea.fr
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00 
#SBATCH --output=/feynman/work/dap/lcs/lg268561/UNIONS/lensfit_matching.log

export OMP_NUM_THREADS=8
source cosmosis-configure
export LD_LIBRARY_PATH=/opt/ohpc/pub/libs/gnu9/gsl/2.7/lib/:$LD_LIBRARY_PATH
cd $WORK/UNIONS/CFIS-UNIONS/CFIS-UNIONS_dev

mpirun --use-hwthread-cpus cosmosis --mpi cosmosis_config/cosmosis_pipeline.ini

# 
# Return exit code
exit 0
