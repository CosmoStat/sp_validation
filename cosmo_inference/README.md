# UNIONS Cosmological Inference Pipeline
by Lisa Goh, CEA Paris-Saclay

This folder contains the files neccessary to run the cosmological inference pipeline on the UNIONS galaxy catalogues. 

### Requirements
To run the pipeline, one would need to have installed [CosmoSIS](https://cosmosis.readthedocs.io/en/latest/) and [CosmoCov](https://github.com/CosmoLike/CosmoCov). 

### To Run
The pipeline requires, as input, the path of the folder containing the TreeCorr xi_plus and xi_minus fits files (assuming they were calculated in the `cosmo_val.py` script).

Start by running the bash script within this folder

```
$ ./pipeline.sh
```

When prompted, specify the path of your 2PCF fits files folder, the desired nz blinding scheme (`A`, `B` or `C`), the path of the folder to store the MCMC chains, as well as your `output_root` name (eg. `shapepipe_v1.0_blind_A`). 

The pipeline starts running automatically, whereby the process can be broken down into several steps:

1. Covariance matrix calculation
    * The covariance matrix is calculated using CosmoCov, by reading in the `./cosmocov_config/cosmocov_{output_root}.ini` file. **Hence make sure the `output_root` here corresponds to the one entered in the initial prompt**.

2. Combining files into CosmoSIS-friendly format
    * This step combines all the ingredients needed by CosmoSIS: the xi_plus/xi_minus fits files calculated from TreeCorr, the covariance matrix, and the nz catalogue into a single `.fits` file to be read in.
    
3. Perform the inference with CosmoSIS
    * This step starts the inference by reading in the `cosmosis_pipeline_{output_root}.ini` file, the `values_{output_root}.ini` file and the `priors_{output_root}.ini` file within the  `./cosmosis_config/` folder. 
4. Submit the job submission bash file to run CosmoSIS on your cluster. Here, an example `submit.sh` script is provided (assuming SLURM architecture, currently running on CEA feynman cluster).

You can finally analyse the chains with the `MCMC.ipynb` notebook. 
