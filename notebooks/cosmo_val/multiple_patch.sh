#!/bin/bash                                                                    
#PBS -k o
### resource allocation
#PBS -l nodes=n08:ppn=24,walltime=48:00:00
### job name
#PBS -N multiple_patch_run
### Redirect stdout and stderr to same file
#PBS -j oe

module purge
module load intelpython3

source activate sp_validation

cd ~/sp_validation/notebooks/cosmo_val

python multiple_patch.py

#cosmosis cosmosis_config/cosmosis_pipeline_SP_v1.3_LFmask_8k.ini


exit 0