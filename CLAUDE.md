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
- Run all tests: `pytest`
- Run fast tests only (recommended for development): `pytest -m "not slow"`
- Run tests with coverage: `pytest --cov=sp_validation --cov-report=term --cov-report=xml`
- Run single test: `pytest src/sp_validation/tests/test_cosmology.py::test_function_name`
- Test performance: Fast tests ~13s, all tests ~18s (vs 30s originally)
- Test tolerances: Optimized based on actual agreement levels and physics constraints

### Linting and Code Quality
- Check code style: `ruff check`
- Auto-fix issues: `ruff check --fix`
- Line length limit: 88 characters

### Installation 
- Install in development mode: `pip install -e .[develop]`
- Install test dependencies: `pip install -e .[test]`

## Architecture

### Core Package Structure (`src/sp_validation/`)
- `basic.py`: Basic utilities and mathematical functions
- `calibration.py`: Shear calibration routines
- `cat.py`: Catalogue handling and manipulation
- `cosmo_val.py`: Cosmology validation routines
- `cosmology.py`: Cosmological calculations and theory
- `galaxy.py`: Galaxy-specific processing
- `io.py`: Input/output utilities
- `plots.py`: Plotting functions
- `rho_tau.py`: Rho and tau statistics calculations
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
Main configuration in `notebooks/params.py` with parameters:
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
Recommended installation via Apptainer/Docker:
```bash
apptainer build --sandbox sp_validation docker://ghcr.io/cosmostat/sp_validation:develop
```

## Notebook Configuration
- The CosmologyValidation class must be initialized in notebooks/cosmo_val