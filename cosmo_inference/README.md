# UNIONS Cosmological Inference Pipeline
by Lisa Goh and Sacha Guerrini, CEA Paris-Saclay

This folder contains the files neccessary to run the cosmological inference pipeline on the UNIONS galaxy catalogues. 

### Requirements
To run the pipeline, one would need to have installed [CosmoSIS](https://cosmosis.readthedocs.io/en/latest/). To sample the PSF leakage parameters, the fork of [cosmosis-standard-library](https://github.com/sachaguer/cosmosis-standard-library/) of Sacha Guerrini has to be used.

Run CosmoSIS with `--mpi`, not `--smp`. The `--smp` process pool is fragile and
barely maintained upstream: its `bcast`, `gather` and `allreduce` methods all
return a `self.data` attribute that is never set, so a run can crash right after
sampling finishes (upstream issue
[cosmosis#170](https://github.com/cosmosis-developers/cosmosis/issues/170) — the
`allreduce` crash was fixed in cosmosis 3.16.1, the rest is still open). Use
cosmosis 3.16.1 or newer, and prefer `--mpi`, which is what the upstream
maintainer recommends.

### To Run
The inference pipeline is orchestrated through Snakemake. On the candide
cluster, drive it with the committed profile — see
[`workflow/README.md`](../workflow/README.md) for the one-time
`uv tool install` setup and the full explanation. From the repository root:

```bash
snakemake --profile workflow/profiles/candide \
    -s workflow/Snakefile \
    inference_fiducial --configfile <run config>
```

Off-cluster, drop `--profile` and add `-j <jobs>` instead. Each job runs
inside the sp_validation container automatically — no `apptainer shell` or
`apptainer exec` needed by hand.

This will automatically execute all steps:
1. Calculate 2PCF ($\xi_{pm}$) via `cosmo_val.py`
2. Compute covariance matrices using CosmoCov <!--- LG: now obsolete, making way for OnCovariance cauclation instead --->
3. Prepare CosmoSIS data (FITS) via `cosmosis_fitting.py`
4. Run CosmoSIS inference

For standalone FITS data preparation (real-space inputs plus optional pseudo-$C_\ell$ data), you can also use the Python script directly:

```bash
python scripts/cosmosis_fitting.py \
  --cosmosis-root "catalog_version_config" \
  --data-dir "/path/to/output/chains" \
  --nz-file "/path/to/nz_file.txt" \
  --out-file "/path/to/output.fits" \
  --xi "/path/to/xi_plus.fits" "/path/to/xi_minus.fits" \
  --cov-xi "/path/to/covariance.txt" \
  --use-rho-tau \
  --rho-stats "/path/to/rho_stats.fits" \
  --tau-stats "/path/to/tau_stats.fits" \
  --cov-tau "/path/to/cov_tau.npy" \
  --cl-file "/path/to/pseudo_cl.fits" \
  --cov-cl "/path/to/pseudo_cl_cov.fits"
```

You can view all available options with:
```bash
python scripts/cosmosis_fitting.py --help
``` 

Ensure the pseudo-$C_\ell$ spectra (`pseudo_cl_*.fits`) and their covariance (`pseudo_cl_cov_*.fits`) produced by `cosmo_val.py` exist for the requested catalog version (or mock seed) before running the standalone command.


This is the pipeline used to derive cosmological constraints with cosmic shear data from the UNIONS v1.4 catalogue.
