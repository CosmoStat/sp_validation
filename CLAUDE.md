# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SP Validation is a Python package for validating weak-lensing catalogues (galaxy and star shapes) produced by ShapePipe. The package performs:

1. **Shear validation**: Calibrates shear catalogues with metacal information, performs PSF leakage tests
2. **Post processing**: Further processing of calibrated shear catalogues  
3. **Cosmology validation**: Diagnostics like rho/tau statistics, E-/B-mode decomposition
4. **Cosmology inference**: Two-point correlation function analysis using CosmoSIS/CosmoCov

## Development Commands

### Testing
Tests live in `src/sp_validation/tests/` and import the full scientific stack,
so run them inside the container.
- Run all tests: `pytest` (collects from `src/sp_validation/tests`; coverage on by default)
- Skip the slow tests: `pytest -m "not slow"`
- Run a single test: `pytest src/sp_validation/tests/test_cosmology.py::test_function_name`

CI runs this same suite inside the freshly-built image before publishing it
(see `.github/workflows/deploy-image.yml`).

### Linting and Code Quality
- Check code style: `ruff check`
- Auto-fix issues: `ruff check --fix`
- Line length limit: 88 characters

### Installation
The package is managed with `uv` (see `uv.lock`); the primary runtime environment
is the container (full scientific stack pre-built). For a local dev environment:
- Dev install (test + docs extras): `uv pip install -e '.[develop]'`
- Test extras only: `uv pip install -e '.[test]'`

## Architecture

### Core Package Structure (`src/sp_validation/`)
- `b_modes.py`: Pure E-/B-mode decomposition (COSEBIS, pseudo-Cℓ)
- `calibration.py`: Shear calibration — the `metacal` response class, galaxy selection masks (size/SNR), and m/c calibration routines
- `cat.py`: Catalogue handling and manipulation
- `cosmo_val.py`: Cosmology validation routines
- `cosmology.py`: Cosmological calculations and theory
- `galaxy.py`: Galaxy-specific processing
- `io.py`: Input/output utilities
- `plots.py`: Plotting functions
- `rho_tau.py`: Rho and tau statistics calculations
- `statistics.py`: Cosmology-independent statistics (jackknife resampling, χ²/PTE, covariance↔correlation, OneCovariance reshaping)
- `survey.py`: Survey-level operations
- `util.py`: General utilities

### External Tools Integration
- **CosmoSIS**: Cosmological inference pipeline (requires separate installation)
- **CosmoCov**: Covariance matrix calculations (requires separate installation)
- **TreeCorr**: Two-point correlation functions
- **Healpy/HealSparse**: Sky map handling

### Cosmology Inference Pipeline (`cosmo_inference/`)
Run via `./pipeline.sh` with flags:
- `--pcf`: Calculate 2-point correlation functions
- `--covmat`: Calculate covariance matrix with CosmoCov
- `--inference`: Run CosmoSIS inference
- `--mcmc_process`: Analyze MCMC chains

### Configuration
Main configuration in `scripts/calibration/params.py` with parameters:
- `name`: Field/patch identifier
- `data_dir`: Input data directory
- `galaxy_cat_path`: Galaxy catalogue path (.fits/.hdf5)
- `star_cat_path`: Star catalogue path (.fits)

### Key Dependencies
- astropy, numpy, scipy for core calculations
- treecorr for correlation functions
- healpy/healsparse for sky maps
- emcee for MCMC sampling
- pyccl for cosmological calculations

## Container Usage
Nothing is hand-built. CI publishes `ghcr.io/cosmostat/sp_validation:<branch>` on
every push, and each person keeps their own copy at
`~/.cache/sp_validation/sp_validation.sif`, managed by the `spv-container` CLI:

```bash
spv-container pull            # fetch :develop there (do it from a compute node)
spv-container status          # which layer is live, and how current it is
spv-container exec <command>  # one-off run inside it
```

Need a package the image lacks mid-analysis? Unpack a writable sandbox once with
`spv-container sandbox`, then `spv-container exec --writable pip install <pkg>`.
The sandbox then takes precedence over the SIF everywhere, workflow jobs
included; `spv-container pull && spv-container sandbox --force` resets it clean.

Every rule runs inside that image, wrapped by Snakemake itself (`--profile
workflow/profiles/candide` on the cluster, `workflow/profiles/default -j N`
elsewhere). The `sp_validation` a rule imports comes from the *launched
checkout*, not the image: `common.configure()` puts its `src/` on the
container's `PYTHONPATH`.

`workflow/README.md` is the full story — profiles, image resolution, refresh.

## Notebook Configuration
- The CosmologyValidation class must be initialized in cosmo_val