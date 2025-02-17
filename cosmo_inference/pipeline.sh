#!/bin/bash

# Transform long options to short ones
for arg in "$@"; do
  shift
  case "$arg" in
    '--help')          set -- "$@" '-h'   ;;
    '--pcf')           set -- "$@" '-p'   ;;
    '--covmat')        set -- "$@" '-c'   ;;
    '--inference')     set -- "$@" '-i'   ;;
    '--mcmc_process')  set -- "$@" '-m'   ;;
    *)                 set -- "$@" "$arg" ;;
  esac
done

# Parse short options
OPTIND=1
while getopts "hpcim" opt
do
  case "$opt" in
    'h') 
        echo "Please input a flag: --help, --pcf, --covmat, --inference or --mcmc_process "; 
        exit 0 
        ;;
    'p') 
        echo "Running cosmo_val.py to calculate 2 point correlation functions";
        python notebooks/cosmo_val/cosmo_val.py
        ;;
    'c') 
        read -p 'ROOT: ' root;
        read -p 'NZ FILE:' nz_file;
        read -p 'PATH COSMOCOV: ' cosmocov;
        echo "Calculating covariance matrices with CosmoCov";
        python scripts/cosmocov_process.py $root $nz_file $cosmocov
        ;;
    'i')
        read -p 'ROOT: ' root;
        read -p 'XI_PLUS/XI_MINUS FITS FILE FOLDER: ' xi_folder;
        read -p 'NZ FILE:' nz_file;
        read -p 'RHO_STATS FILE FOLDER: ' rho_stats_folder;
        read -p 'USE TAU_STATS? (y/n): ' tau_stats;
        read -p 'COV_XI MAT TXT FILE:' covmat;
        read -p 'OUTPUT MCMC CHAIN FOLDER: ' data;
        
        out_file="data/${root}/cosmosis_${root}.fits";
        
        #LG: add check if xi_plus/xi_minus fits file exists
        python scripts/cosmosis_fitting.py $root $xi_folder $covmat $nz_file $rho_stats_folder $out_file $tau_stats;

        sed -i "/^\[DEFAULT\]/a\FITS_FILE = ${out_file}" cosmosis_config/cosmosis_pipeline_${root}.ini;
        sed -i "/^\[output\]/a\filename = ${data}/samples_${root}.txt" cosmosis_config/cosmosis_pipeline_${root}.ini;
        sed -i "/^\[pipeline\]/a\values = cosmosis_config/values_${root}.ini" cosmosis_config/cosmosis_pipeline_${root}.ini;
        sed -i "/^\[pipeline\]/a\priors = cosmosis_config/priors_${root}.ini" cosmosis_config/cosmosis_pipeline_${root}.ini;
        
        echo "Prepared CosmoSIS configuration file in cosmosis_config/cosmosis_pipeline_${root}.ini";
        echo "You can now run the inference with the command: cosmosis cosmosis_config/cosmosis_pipeline_${root}.ini"
        ;;
    'm')
        # LG: also convert this into a script to directly output contour plots
        echo "Run the cosmo_inference/notebooks/MCMC.ipynb notebook to analyse your chains"
        ;;
    '?') 
        print_usage >&2; 
        exit 1 
        ;;
  esac
done
shift $(expr $OPTIND - 1) # remove options from positional parameters