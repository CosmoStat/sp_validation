# sp_validation

Validational of weak-lensing catalogues (galaxy and star shapes and other parameters) produced by [ShapePipe](https://github.com/CosmoStat/shapepipe).

| Usage | Development | Release |
| ----- | ----------- | ------- |
| [![docs](https://img.shields.io/badge/docs-Sphinx-blue)](https://martin.kilbinger.github.io/sp_validation/) | [![build](https://github.com/martin.kilbinger/sp_validation/workflows/CI/badge.svg)](https://github.com/martin.kilbinger/sp_validation/actions?query=workflow%3ACI) | [![release](https://img.shields.io/github/v/release/martin.kilbinger/sp_validation)](https://github.com/martin.kilbinger/sp_validation/releases/latest) |
| [![license](https://img.shields.io/github/license/martin.kilbinger/sp_validation)](https://github.com/martin.kilbinger/sp_validation/blob/master/LICENCE.txt) | [![deploy](https://github.com/martin.kilbinger/sp_validation/workflows/CD/badge.svg)](https://github.com/martin.kilbinger/sp_validation/actions?query=workflow%3ACD) | [![pypi](https://img.shields.io/pypi/v/sp_validation)](https://pypi.org/project/sp_validation/) |
| [![wemake-python-styleguide](https://img.shields.io/badge/style-wemake-000000.svg)](https://github.com/wemake-services/wemake-python-styleguide) | [![codecov](https://codecov.io/gh/martin.kilbinger/sp_validation/branch/master/graph/badge.svg?token=XHJIQXV7AX)](https://codecov.io/gh/martin.kilbinger/sp_validation) | [![python](https://img.shields.io/pypi/pyversions/sp_validation)](https://www.python.org/downloads/source/) |
| [![contribute](https://img.shields.io/badge/contribute-read-lightgrey)](https://github.com/martin.kilbinger/sp_validation/blob/master/CONTRIBUTING.md) | [![CodeFactor](https://www.codefactor.io/repository/github/martin.kilbinger/sp_validation/badge)](https://www.codefactor.io/repository/github/martin.kilbinger/sp_validation) | |
| [![coc](https://img.shields.io/badge/conduct-read-lightgrey)](https://github.com/martin.kilbinger/sp_validation/blob/master/CODE_OF_CONDUCT.md) | [![Updates](https://pyup.io/repos/github/martin.kilbinger/sp_validation/shield.svg)](https://pyup.io/repos/github/martin.kilbinger/sp_validation/) | |

---
> Author: <a href="www.cosmostat.org/people/kilbinger" target="_blank" style="text-decoration:none; color: #F08080">Axel Guinot, Martin Kilbinger, Samuel Farrens, Emma Ayçoberry</a>  
> Email: <a href="mailto:samuel.farrens@cea.fr" style="text-decoration:none; color: #F08080">axel.guinot.astro@gmail.com</a>  
> Year: 2021  
---

See [pyraliddemo](https://github.com/sfarrens/pyraliddemo) for a demo package created with the Pyralid template.

## Contents

## Run the validation notebooks

### Set up

Edit the file `notebooks/params.py` according to your data.

Make sure that all input files set in `params.py` are accessible from the run directory.
The run directory needs to contain all files in `notebook`, i.e. `params.py` and all `.ipynb` notebooks

### Run

1. In the run directory start JupyterLab:
  ```bash
  jupyer-lab
  ```

2. Load and run first notebook `main_set_up.ipynb`:
   1. Double-click in file manager tab on the left.
   2. Run notebook.

3. Run all further notebooks:
   1. Double-click as above.
   2. Change kernel to `main_set_up.ipynb`:
      Either via the menu `Kernel -> Change Kernel` or
      by clicking on `Python 3` on the top left of the notebook.
   3. Run notebook.

Run the notebooks in the following order:
   1. `main_set_up.ipynb` (main notebook)
   2. `metacal_global.ipynb`
   3. [`metacal_local.ipynb`] optional     
   4. `psf_leakage.ipynb'
   5. `write_cat.ipynb`
   6. `cosmology.ipynb`


