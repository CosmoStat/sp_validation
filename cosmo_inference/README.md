# UNIONS Cosmological Inference Pipeline
by Lisa Goh, CEA Paris-Saclay

This folder contains the files neccessary to run the cosmological inference pipeline on the UNIONS galaxy catalogues. 

### Requirements
To run the pipeline, one would need to have installed [CosmoSIS](https://cosmosis.readthedocs.io/en/latest/) and [CosmoCov](https://github.com/CosmoLike/CosmoCov). 

### To Run
Run the bash script within this folder

```
$ ./pipeline.sh
```
with one of the following flags:

`--pcf`: This step runs the `cosmo_val.py` script to calculate the various 2 point correlation functions. It will also write the $\xi_{pm}$ correlation functions in a fits file. 

`--covmat`: The covariance matrix is calculated here using CosmoCov, by reading in the `./cosmocov_config/cosmocov_{output_root}.ini` file. **Hence make sure the `output_root` here corresponds to the one entered in the prompt**.

`--inference`: This step writes out the relevant `./cosmosis_config/cosmosis_{output_root}.ini` file, in order to run CosmoSIS to conduct the cosmological inference. It also combines the data needed by CosmoSIS: the $\xi_{pm}$ fits files calculated in `cosmo_val.py`, the covariance matrix, and the nz catalogue, into a single `.fits` file. It also fetches the rho-statistics computed in `cosmo_val.py` to marginalize on PSF leakage parameters. You can submit the job submission bash script to run CosmoSIS on your cluster. Here, an example `submit.sh` script is provided (assuming SLURM architecture, currently running on CEA feynman cluster).

`--mcmc_process`: You can finally analyse the chains with the `MCMC.ipynb` notebook. 


