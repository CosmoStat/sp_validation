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
        read -p 'COSMOSIS ROOT: ' cosmosis_root;

        # Create the folder if it does not exist
        if [ ! -d "data/$cosmosis_root" ]; then
            mkdir -p "data/$cosmosis_root"
            echo "Directory 'data/$cosmosis_root' created."
        else
            echo "Directory 'data/$cosmosis_root' already exists."
        fi

        read -p 'COSMO_VAL OUTPUT FOLDER: ' output_folder;
        read -p 'NZ FILE:' nz_file;
        read -p 'OUTPUT MCMC CHAIN FOLDER: ' data;
        read -p 'USE PSEUDO_CELL? (y/n): ' pseudo_cell;

        if [ "${pseudo_cell}" == "y" ]; then

          read -p 'CL ROOT: ' cl_root;
          read -p 'GAUSSIAN COVARIANCE? (iNKA/OneCovariance): ' gaussian_part;


          echo "Using pseudo cell"
          out_file="data/${cosmosis_root}/cosmosis_${cosmosis_root}_cell.fits"

          python scripts/cosmosis_fitting.py $cosmosis_root $output_folder $nz_file $pseudo_cell $out_file $cl_root $gaussian_part;
        else
          read -p 'XI ROOT: ' xi_root;
          read -p 'USE RHO/TAU_STATS? (y/n): ' rhotau_stats;
          echo $rhotau_stats
          read -p 'TAU ROOT: ' tau_root;
          read -p 'COV_XI MAT TXT FILE:' covmat;
          
          out_file="data/${cosmosis_root}/cosmosis_${cosmosis_root}.fits";

          #LG: add check if xi_plus/xi_minus fits file exists
          python scripts/cosmosis_fitting.py $cosmosis_root $output_folder $nz_file $pseudo_cell $out_file $xi_root $rhotau_stats $tau_root $covmat;

        fi
        
        if [ "${pseudo_cell}" == "y" ]; then
          output_ini_file="cosmosis_config/cosmosis_pipeline_${cosmosis_root}_cell.ini"
          cp cosmosis_config/cosmosis_pipeline_A_ia_cell.ini $output_ini_file
        else
          output_ini_file="cosmosis_config/cosmosis_pipeline_${cosmosis_root}.ini"
          if [ "${rhotau_stats}" == "y" ]; then
            cp cosmosis_config/cosmosis_pipeline_A_psf.ini $output_ini_file;
          else
            cp cosmosis_config/cosmosis_pipeline_A_ia.ini $output_ini_file;
          fi
        fi

        sed -i "/^\[DEFAULT\]/a\SCRATCH = ${data}" $output_ini_file;
        sed -i "/^\[DEFAULT\]/a\FITS_FILE = ${out_file}" $output_ini_file;
        if [ "${pseudo_cell}" == "y" ]; then
          sed -i "/^\[output\]/a\filename = %(SCRATCH)s/${cosmosis_root}_cell/samples_${cosmosis_root}_cell.txt" $output_ini_file;
          sed -i "/^\[pipeline\]/a\values = cosmosis_config/values_ia.ini" $output_ini_file;
          sed -i "/^\[pipeline\]/a\priors = cosmosis_config/priors.ini" $output_ini_file;
          sed -i "/^\[2pt_like]/a\file = %(COSMOSIS_DIR)s/likelihood/2pt/2pt_like.py" $output_ini_file;
          sed -i "/^\[2pt_like]/a\data_sets=CELL_EE" $output_ini_file;
          sed -i "/^\[polychord\]/a\polychord_outfile_root = ${cosmosis_root}_cell" $output_ini_file;
          sed -i "/^\[test\]/a\save_dir = %(SCRATCH)s/best_fit/${cosmosis_root}_cell" $output_ini_file;
        else
          sed -i "/^\[output\]/a\filename = %(SCRATCH)s/${cosmosis_root}/samples_${cosmosis_root}.txt" $output_ini_file;
          if [ "${rhotau_stats}" == "y" ]; then
            sed -i "/^\[pipeline\]/a\values = cosmosis_config/values_psf.ini" $output_ini_file;
            sed -i "/^\[pipeline\]/a\priors = cosmosis_config/priors_psf.ini" $output_ini_file;
            sed -i "/^\[2pt_like]/a\file = %(COSMOSIS_DIR)s/likelihood/2pt/2pt_like_xi_sys.py" $output_ini_file;
            sed -i "/^\[2pt_like]/a\data_sets=XI_PLUS XI_MINUS TAU_0_PLUS TAU_2_PLUS" $output_ini_file;
            sed -i "/^\[2pt_like]/a\add_xi_sys=T" $output_ini_file;
          else
            sed -i "/^\[pipeline\]/a\values = cosmosis_config/values_ia.ini" $output_ini_file;
            sed -i "/^\[pipeline\]/a\priors = cosmosis_config/priors.ini" $output_ini_file;
            sed -i "/^\[2pt_like]/a\file = %(COSMOSIS_DIR)s/likelihood/2pt/2pt_like.py" $output_ini_file;
            sed -i "/^\[2pt_like]/a\data_sets=XI_PLUS XI_MINUS" $output_ini_file;
          fi
          sed -i "/^\[polychord\]/a\polychord_outfile_root = ${root}" $output_ini_file;
          sed -i "/^\[test\]/a\save_dir = %(SCRATCH)s/best_fit/${root}" $output_ini_file;
        fi
        echo "Prepared CosmoSIS configuration file in $output_ini_file";
        echo "You can now run the inference with the command: cosmosis $output_ini_file"
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