# sp_validation

Validation of weak-lensing catalogues (galaxy and star shapes and other parameters) produced by [ShapePipe](https://github.com/CosmoStat/shapepipe).

[![docs](https://img.shields.io/badge/docs-sphinx-blue)](https://cosmostat.github.io/sp_validation/)
[![CI](https://github.com/CosmoStat/sp_validation/actions/workflows/deploy-image.yml/badge.svg)](https://github.com/CosmoStat/sp_validation/actions/workflows/deploy-image.yml)
[![container](https://img.shields.io/badge/container-ghcr.io-2496ED?logo=docker&logoColor=white)](https://github.com/CosmoStat/sp_validation/pkgs/container/sp_validation)
[![python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-blue)](https://github.com/CosmoStat/sp_validation/blob/develop/LICENCE.txt)
[![code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![contribute](https://img.shields.io/badge/contribute-read-lightgrey)](https://github.com/CosmoStat/sp_validation/blob/develop/CONTRIBUTING.md)
[![code of conduct](https://img.shields.io/badge/conduct-read-lightgrey)](https://github.com/CosmoStat/sp_validation/blob/develop/CODE_OF_CONDUCT.md)

---
> **Authors:** the [CosmoStat](https://www.cosmostat.org) lab at CEA Paris-Saclay — Martin Kilbinger, Cail Daley, Sacha Guerrini.  
> **Contributors:** Emma Ayçoberry, Lucie Baumont, Clara Bonini, Samuel Farrens, Lisa Goh, Axel Guinot, Fabian Hervas Peters.  
> **Contact:** [martin.kilbinger@cea.fr](mailto:martin.kilbinger@cea.fr)
---

This package contains a library and several scripts and notebooks. The main
tasks that can be performed by `sp_validation` are:
- Shear validation, in particular for the output of the `shapepipe`
  pipeline. This task takes on input a shear catalogue with metacal information,
  performs the calibration and carries out various tests, e.g. PSF leakage.
  A calibrated shear catalogue is then created on output.  
- Post processing. A number of scripts allow further processing of the above
  output calibrated shear catalogue.  
- Cosmology validation. This task uses the calibrated shear catalogue from
  above to run detailed diagnostics useful for further cosmology analysis,
  e.g. rho- and tau-statistics, E-/B-mode decomposition. Several catalogues
  can be compared and useful plots are created.
- Cosmology inference. This task uses the calibrated shear catalogue from
  a shear validation run and performes cosmology inference using the two-point
  correlation function.

## Repository layout

```
src/sp_validation/    library code (incl. glass_mock core)
cosmo_val/            validation: code + config        (promoted from notebooks/)
cosmo_inference/      inference: code + config         (cosmosis / cosmocov)
workflow/             ALL analysis — modular Snakemake, multi-person → results/
papers/               final-figure assembly only (PDF, colour, layout)
scripts/              real reduction + runner scripts (catalog builders, masking, glass-mock runners)
scratch/              per-person — ad hoc work + personal workflows (tracked)
results/              analysis products + diagnostic plots (contents gitignored, dir kept)
docs/  tests/  config/
```

The dividing line is the **inputs to a paper figure**. Everything up to that
point is *analysis* and lives in `workflow/`: generic, reusable, modular
Snakemake, organized for several people, producing both products and diagnostic
plots into a single top-level `results/`. The figure itself is *presentation*
and lives in `papers/<paper>/`: final-figure assembly only — PDF, colour, layout
— tied to one paper, and free to never touch Snakemake. `scratch/<person>/` is
personal and ad hoc, tracked because sharing scratch is useful. `cosmo_val/` and
`cosmo_inference/` are the side-by-side code+config homes for the validation and
inference stages.

The workflow scales by being modular, not monolithic: Snakemake's `module`
directive imports the shared rules under each run's own config and an output
`prefix`, so runs namespace under `results/<name>/` without clobbering one
another.

## Container Installation (Recommended)

The easiest way to install sp_validation is via a container. Docker images are automatically built and pushed to the [GitHub Container Registry (GHCR)](https://github.com/CosmoStat/sp_validation/pkgs/container/sp_validation) on every push to `develop`. This image can be installed and run on most systems (including clusters) with just a few lines of code.

We recommend running the image with **Apptainer** (formerly Singularity) which is installed on most HPC clusters. To simply run the image, use the following command:

```bash
# build writeable "sandbox" container in the current directory
# ./sp_validation will be a directory that functions like a vm
apptainer build --sandbox sp_validation docker://ghcr.io/cosmostat/sp_validation:develop

# open a shell in the container
apptainer shell --writable sp_validation 
# and confirm that the installation was successful
python -c "import sp_validation"
```

You can also run the image with **Docker**:

```bash
docker run --rm -it ghcr.io/cosmostat/sp_validation:develop python -c "import sp_validation"
```

We do not currently build images for Apple Silicon/arm64; however the amd64 images should work on these systems, albeit with reduced performance.

## Local Installation

Requires Python ≥ 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
uv pip install -e '.[test]'
```

## Flow chart

The following flow chart illustrates the steps required to go from ShapePipe output products
to calibrated and well-selected galaxy catalogues.

![Flow chart](docs/images/flow_chart.png)


## Run shear validation

See the [documentation](docs/source/run_validation.md) for instructions on how to set up and run `sp_validation`.


## Post processing

The output(s) of one or more [shear validation runs](#run-shear-validation) can
be processed further with post-processing scripts. See
[here](docs/source/post_processing.md) for details.

## Cosmology validation

TBD.

## Cosmology inference

See the corresponding [documentation](cosmo_inference/README.md).
