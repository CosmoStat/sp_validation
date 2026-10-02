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

The dormant `inference_fiducial` target declares a CosmoSIS FITS data file and
`.ini`; it does not launch the sampler. Its pseudo-$C_\ell$ data-vector input is
an unproduced FITS path: the `pseudo_cl` rule writes a SACC part, while
`pseudo_cl_cov` does write FITS.

For standalone FITS data preparation (real-space inputs plus optional pseudo-$C_\ell$ data), you can also use the Python script directly:

```bash
python scripts/cosmosis_fitting.py \
  --cosmosis-root "catalog_version_config" \
  --data-dir "/path/to/output/chains" \
  --nz-file "/path/to/nz_file.txt" \
  --output-root "/path/to/output" \
  --output-basename "catalog_version_config" \
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

The `pseudo_cl` rule writes the data vector as a SACC part, while
`cosmosis_fitting.py` expects a pseudo-$C_\ell$ FITS input. Its covariance rule
writes FITS, but the requested data-vector FITS path has no producer; supply a
compatible FITS file separately for standalone use.

The published UNIONS v1.4 chains used a separate configuration. The committed
templates have different fiducial priors for $m_1$, $\Delta z$, $\alpha$,
$A_\mathrm{IA}$, and $\Omega_b$.
