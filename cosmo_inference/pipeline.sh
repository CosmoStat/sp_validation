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
        read -p 'COVARIANCE FILE: ' covmat_file;
        read -p 'OUTPUT STUB (without extension): ' output_stub;
        echo "Processing covariance matrix";
        python scripts/cosmocov_process.py $covmat_file $output_stub
        ;;
    'i')
        read -p 'XI ROOT: ' xi_root;
        read -p 'TAU ROOT: ' tau_root;
        read -p 'COSMOSIS ROOT: ' cosmosis_root;
        read -p 'COSMO_VAL OUTPUT FOLDER: ' output_folder;
        read -p 'NZ FILE:' nz_file;
        read -p 'USE RHO/TAU_STATS? (y/n): ' rhotau_stats;
        echo $rhotau_stats
        read -p 'COV_XI MAT TXT FILE:' covmat;
        read -p 'OUTPUT MCMC CHAIN FOLDER: ' data;
        
        # Use cosmosis_root for output file naming
        out_file="data/${cosmosis_root}/cosmosis_${cosmosis_root}.fits";
        
        #LG: add check if xi_plus/xi_minus fits file exists
        python scripts/cosmosis_fitting.py $xi_root $tau_root $output_folder $covmat $nz_file $out_file $rhotau_stats;
        
        if [ "${rhotau_stats}" == "y" ]; then
          cp cosmosis_config/cosmosis_pipeline_A_psf.ini cosmosis_config/cosmosis_pipeline_${cosmosis_root}.ini;
        else
          cp cosmosis_config/cosmosis_pipeline_A_ia.ini cosmosis_config/cosmosis_pipeline_${cosmosis_root}.ini;
        fi

        sed -i "/^\[DEFAULT\]/a\SCRATCH = ${data}" cosmosis_config/cosmosis_pipeline_${cosmosis_root}.ini;
        sed -i "/^\[DEFAULT\]/a\FITS_FILE = ${out_file}" cosmosis_config/cosmosis_pipeline_${cosmosis_root}.ini;
        sed -i "/^\[output\]/a\filename = %(SCRATCH)s/${cosmosis_root}/samples_${cosmosis_root}.txt" cosmosis_config/cosmosis_pipeline_${cosmosis_root}.ini;
        if [ "${rhotau_stats}" == "y" ]; then
          sed -i "/^\[pipeline\]/a\values = cosmosis_config/values_psf.ini" cosmosis_config/cosmosis_pipeline_${cosmosis_root}.ini;
          sed -i "/^\[pipeline\]/a\priors = cosmosis_config/priors_psf.ini" cosmosis_config/cosmosis_pipeline_${cosmosis_root}.ini;
          sed -i "/^\[2pt_like]/a\file = %(COSMOSIS_DIR)s/likelihood/2pt/2pt_like_xi_sys.py" cosmosis_config/cosmosis_pipeline_${cosmosis_root}.ini;
          sed -i "/^\[2pt_like]/a\data_sets=XI_PLUS XI_MINUS TAU_0_PLUS TAU_2_PLUS" cosmosis_config/cosmosis_pipeline_${cosmosis_root}.ini;
          sed -i "/^\[2pt_like]/a\add_xi_sys=T" cosmosis_config/cosmosis_pipeline_${cosmosis_root}.ini;
        else
          sed -i "/^\[pipeline\]/a\values = cosmosis_config/values_ia.ini" cosmosis_config/cosmosis_pipeline_${cosmosis_root}.ini;
          sed -i "/^\[pipeline\]/a\priors = cosmosis_config/priors.ini" cosmosis_config/cosmosis_pipeline_${cosmosis_root}.ini;
          sed -i "/^\[2pt_like]/a\file = %(COSMOSIS_DIR)s/likelihood/2pt/2pt_like.py" cosmosis_config/cosmosis_pipeline_${cosmosis_root}.ini;
          sed -i "/^\[2pt_like]/a\data_sets=XI_PLUS XI_MINUS" cosmosis_config/cosmosis_pipeline_${cosmosis_root}.ini;
        fi
        sed -i "/^\[polychord\]/a\polychord_outfile_root = ${cosmosis_root}" cosmosis_config/cosmosis_pipeline_${cosmosis_root}.ini;
        
        echo "Prepared CosmoSIS configuration file in cosmosis_config/cosmosis_pipeline_${cosmosis_root}.ini";
        echo "You can now run the inference with the command: cosmosis cosmosis_config/cosmosis_pipeline_${cosmosis_root}.ini"
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