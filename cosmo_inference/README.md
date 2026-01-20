# UNIONS Cosmological Inference Pipeline
by Lisa Goh and Sacha Guerrini, CEA Paris-Saclay

This folder contains the files neccessary to run the cosmological inference pipeline on the UNIONS galaxy catalogues. 

### Requirements
To run the pipeline, one would need to have installed [CosmoSIS](https://cosmosis.readthedocs.io/en/latest/) and [CosmoCov](https://github.com/CosmoLike/CosmoCov). To PSF leakage parameters, the fork of [cosmosis-standard-library](https://github.com/sachaguer/cosmosis-standard-library/) of Sacha Guerrini has to be used.

### To Run
Run the bash script within this folder

```
$ ./pipeline.sh
```
with one of the following flags:

`--pcf`: This step runs the `cosmo_val.py` script to calculate the various 2 point correlation functions. It will also write the $\xi_{pm}$ correlation functions in a fits file. 

`--covmat`: The covariance matrix is calculated here using CosmoCov, by reading in the `./cosmocov_config/cosmocov_{output_root}.ini` file. **Hence make sure the `output_root` here corresponds to the one entered in the prompt**.

`--inference`: This step writes out the relevant `./cosmosis_config/cosmosis_{output_root}.ini` file, in order to run CosmoSIS to conduct the cosmological inference. It also combines the data needed by CosmoSIS: the $\xi_{pm}$ fits files calculated in `cosmo_val.py`, the covariance matrix, and the nz catalogue, into a single `.fits` file. It also fetches the rho-statistics computed in `cosmo_val.py` to marginalize on PSF leakage parameters. To do so, specify the path to the `cosmo_val` outputs. Note that the Rho- and Tau-statistics have to be stored in a folder `rho_tau_stats` in this folder. If the 2PCF $\xi_{pm}$ are not found it will raise an `Error` to run `cosmo_val`. If the Rho- and Tau-statistics do not exist, the script will raise a   `Warning` to run `cosmo_val` but will create a data vector without the Rho- and Tau-statistics.

`--mcmc_process`: You can finally analyse the chains with the `MCMC.ipynb` notebook. 


This is the pipeline used to derive cosmological constraints with cosmic shear data from the UNIONS v1.4 catalogue.