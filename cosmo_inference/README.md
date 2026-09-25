# UNIONS Cosmological Inference Pipeline
by Lisa Goh and Sacha Guerrini, CEA Paris-Saclay

This folder contains the files neccessary to run the cosmological inference pipeline on the UNIONS galaxy catalogues. 

### Requirements
Everything the pipeline needs ships in the container: nothing to install, and no
paths to edit before a run.

[CosmoSIS](https://cosmosis.readthedocs.io/en/latest/) comes in via the
`workflow` extra, built with MPI support. The CosmoSIS Standard Library — the
tree of modules the `.ini` pipelines name — is built into the image at
`/opt/cosmosis-standard-library`, with `CSL_DIR` pointing there. CosmoSIS reads
environment variables into an `.ini`'s `[DEFAULT]` section, so the templates'
`COSMOSIS_DIR = %(CSL_DIR)s` resolves to it. Outside the container, export
`CSL_DIR` at a build of your own and the same templates work unchanged.

CSL is pinned to the **UNIONS-WL org fork**
([UNIONS-WL/cosmosis-standard-library](https://github.com/UNIONS-WL/cosmosis-standard-library/))
at `b7b1552a`, the fork's main branch: Sacha's four UNIONS commits are
reapplied on current upstream, including the scipy `lpn` fix.

Launch sampling under MPI (`mpiexec -n N cosmosis --mpi ...`), not `--smp`:
CosmoSIS's shared-memory pool is unmaintained and still crashes after sampling
completes (`Pool` has no attribute `data`, `runtime/process_pool.py`) as of
3.25.2.

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
