!/bin/bash
read -p 'FITS FILE FOLDER: ' xi_folder
read -p 'NZ BLIND:' blind
read -p 'CHAIN FOLDER: ' data
read -p 'ROOT: ' root
mkdir -p data/${root}

################ STEP 1: CALCULATE COVARIANCE MATRICES###############
# make a folder to store the covariances
mkdir -p data/${root}/covs

nz_file="data/nz/nz_shapepipe_${blind}.txt"

# write to the ini file, paths to the nz file
cat > cosmocov_config/cosmocov_${root}.ini <<EOF

shear_REDSHIFT_FILE : $nz_file 
clustering_REDSHIFT_FILE : $nz_file
outdir : data/$root/covs/

EOF

echo -e "Running CosmoCov...\n"

# run cosmocov (SPECIFY LOCATION OF COSMOCOV LIBRARY)
for i in {1..3}; 
    do /feynman/work/dap/lcs/lg268561/UNIONS/CFIS-UNIONS/CosmoCov/covs/cov $i cosmocov_config/cosmocov_${root}.ini; 
done

# do postprocessing (plot covmat and write into txt file)
f="data/${root}/covs/cov_${root}"; cat data/${root}/covs/out_cov* > $f; python scripts/cosmocov_process.py $f

echo -e "Covmats written out to data/${root}/covs/cov_${root}\n"
################## STEP 2: COMBINE##########################################################
echo -e "Combining files...\n"
# Specify paths to the xi_plus and xi_minus fits files, and the covmat txt file
xip_cat="${xi_folder}/xiplus_shapepipe.fits"
xim_cat="${xi_folder}/ximinus_shapepipe.fits"
covmat="data/${root}/covs/cov_${root}.txt"

out_file="data/${root}/cosmosis_${root}.fits"

python scripts/cosmosis_fitting.py $xip_cat $xim_cat $covmat $nz_file $out_file

################# STEP 3: RUN COSMOSIS#####################################################
# Add path specification within cosmosis config file
sed "/^\[DEFAULT\]/a\FITS_FILE = ${out_file}" cosmosis_config/cosmosis_pipeline_${root}.ini
sed "/^\[output\]/a\filename = ${data}/samples_${root}.txt" cosmosis_config/cosmosis_pipeline_${root}.ini
sed "/^\[pipeline\]/a\values = cosmosis_config/values_${root}.ini" cosmosis_config/cosmosis_pipeline_${root}.ini
sed "/^\[pipeline\]/a\priors = cosmosis_config/priors_${root}.ini" cosmosis_config/cosmosis_pipeline_${root}.ini

echo -e "Running CosmoSIS...\n"

# Submit cosmosis job to run on cluster, specifying your output log file path
sbatch -J unions_${root} --output=$WORK/UNIONS/cfis_${root}.log scripts/submit.sh

echo -e "JOB SUBMITTED\n"
echo -e "-------------PIPELINE END----------------"
