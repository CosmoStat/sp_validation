#!/bin/bash
read -p 'SHEAR CATALOGUE: ' shear_cat
# read -p 'NZ CATALOGUE: ' nz_cat
read -p 'DATA ROOT: ' root
read -p 'OUT ROOT: ' out_root
# read -p 'BLIND:' blind
mkdir -p data/${root}

# #################STEP 0: RUN NOTEBOOK TO ANALYSE CATALOGUE; DERIVE PLOTS##################
# #File: cfis_analysis.ipynb

# ##################STEP 1: CALCULATE XIP/XIM (OUTPUTS TREECORR FITS CATALOG)###############
python treecorr_calc.py $shear_cat $root

echo -e "2PCF's calculated!\n"

# # ##################STEP 2: WRITE NZ's######################################################
# python nz_writeout.py $nz_cat $root $blind

# echo -e "nz's written out!\n"

# # # # ##################STEP 3: ESTIMATE COVMATS################################################

# # #edit ini file 
# mkdir -p data/${root}/covs

# nz_file="data/${root}/nz_shapepipe_A.txt"

# # sed -i  "/shear_REDSHIFT_FILE/c shear_REDSHIFT_FILE : $nz_file" cosmocov.ini
# # sed -i  "/clustering_REDSHIFT_FILE/c clustering_REDSHIFT_FILE : $nz_file" cosmocov.ini
# # sed -i "/outdir/c outdir : data/$root/covs/" cosmocov.ini

# echo -e "Running CosmoCov...\n"

# ##run cosmocov
# for i in {1..3}; 
#     do ../CosmoCov/covs/cov $i cosmocov.ini; 
# done

# # do postprocessing (plot covmat and write into txt file)
# f="data/${root}/covs/cov_${root}"; cat data/${root}/covs/out_cov* > $f; python cosmocov_process.py $f

# # # # # ##################STEP 4: COMBINE##########################################################
# xip_cat="data/${root}/xiplus_${root}.fits"
# xim_cat="data/${root}/ximinus_${root}.fits"
# covmat="data/${root}/covs/cov_${root}.txt"

# out_file="$PWD/data/${root}/cosmosis_${root}.fits"

python cosmosis_fitting.py /n23data1/n06data/lgoh/scratch/CFIS-UNIONS/CFIS-UNIONS_dev/cosmo_inference/data/SP_v1.4_A/xiplus_SP_v1.4_A.fits /n23data1/n06data/lgoh/scratch/CFIS-UNIONS/CFIS-UNIONS_dev/cosmo_inference/data/SP_v1.4_A/ximinus_SP_v1.4_A.fits /n23data1/n06data/lgoh/scratch/CFIS-UNIONS/CFIS-UNIONS_dev/cosmo_inference/data/SP_v1.4_A/covs/cov_SP_v1.4.txt /n23data1/n06data/lgoh/scratch/CFIS-UNIONS/CFIS-UNIONS_dev/cosmo_inference/data/nz/nz_shapepipe_A.txt /n23data1/n06data/lgoh/scratch/CFIS-UNIONS/CFIS-UNIONS_dev/cosmo_inference/data/SP_v1.4_A/cosmosis_SP_v1.4_A.fits

# # # # ##################STEP 5: RUN COSMOSIS#####################################################
# echo -e "Running CosmoSIS...\n"

# sed -i "/SCRATCH = /c SCRATCH = $WORK/UNIONS/chains/${out_root}/" cosmosis_config/cosmosis_pipeline.ini
# sed -i "/FITS_FILE = /c FITS_FILE = ${out_file}" cosmosis_config/cosmosis_pipeline.ini
# sed -i "/filename = /c filename = %(SCRATCH)s/samples_${out_root}.txt" cosmosis_config/cosmosis_pipeline.ini

# #submit cosmosis job to run on cluster

# sbatch -J cfis_${root} --output=$WORK/UNIONS/cfis_${out_root}.log slurm.sh

# echo -e "-------------PIPELINE END----------------"

# # ##################STEP 6: RUN NOTEBOOK TO ANALYSE CONTOURS (WITH GETDIST)##################
# #File: CFIS_plotting.ipynb