# sp_validation

Validation of weak-lensing catalogues (galaxy and star shapes and other parameters) produced by [ShapePipe](https://github.com/CosmoStat/shapepipe).

| Usage | Development | Release |
| ----- | ----------- | ------- |
| [![docs](https://img.shields.io/badge/docs-Sphinx-blue)](https://martin.kilbinger.github.io/sp_validation/) | [![build](https://github.com/martin.kilbinger/sp_validation/workflows/CI/badge.svg)](https://github.com/martin.kilbinger/sp_validation/actions?query=workflow%3ACI) | [![release](https://img.shields.io/github/v/release/martin.kilbinger/sp_validation)](https://github.com/martin.kilbinger/sp_validation/releases/latest) |
| [![license](https://img.shields.io/github/license/martin.kilbinger/sp_validation)](https://github.com/martin.kilbinger/sp_validation/blob/master/LICENCE.txt) | [![deploy](https://github.com/martin.kilbinger/sp_validation/workflows/CD/badge.svg)](https://github.com/martin.kilbinger/sp_validation/actions?query=workflow%3ACD) | [![pypi](https://img.shields.io/pypi/v/sp_validation)](https://pypi.org/project/sp_validation/) |
| [![wemake-python-styleguide](https://img.shields.io/badge/style-wemake-000000.svg)](https://github.com/wemake-services/wemake-python-styleguide) | [![codecov](https://codecov.io/gh/martin.kilbinger/sp_validation/branch/master/graph/badge.svg?token=XHJIQXV7AX)](https://codecov.io/gh/martin.kilbinger/sp_validation) | [![python](https://img.shields.io/pypi/pyversions/sp_validation)](https://www.python.org/downloads/source/) |
| [![contribute](https://img.shields.io/badge/contribute-read-lightgrey)](https://github.com/martin.kilbinger/sp_validation/blob/master/CONTRIBUTING.md) | [![CodeFactor](https://www.codefactor.io/repository/github/martin.kilbinger/sp_validation/badge)](https://www.codefactor.io/repository/github/martin.kilbinger/sp_validation) | |
| [![coc](https://img.shields.io/badge/conduct-read-lightgrey)](https://github.com/martin.kilbinger/sp_validation/blob/master/CODE_OF_CONDUCT.md) | [![Updates](https://pyup.io/repos/github/martin.kilbinger/sp_validation/shield.svg)](https://pyup.io/repos/github/martin.kilbinger/sp_validation/) | |

---
> Author: <a href="www.cosmostat.org/people/kilbinger" target="_blank" style="text-decoration:none; color: #F08080">Axel Guinot, Martin Kilbinger, Samuel Farrens, Emma Ayçoberry</a>  
> Email: <a href="mailto:martin.kilbinger@cea.fr" style="text-decoration:none; color: #F08080">martin.kilbinger@cea.fr</a>  
> Year: 2021  
---

See [pyraliddemo](https://github.com/sfarrens/pyraliddemo) for a demo package created with the Pyralid template.


## Run validation

See the [documentation](docs/source/run_validation.md).

### Run

There are two possibilities to carry out the validation, by running the jupyter notebooks
in a browser, or by running a python script in the command line via `ipython`.

#### Running the jupyter notebooks

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
   4. `psf_leakage.ipynb`
   5. `write_cat.ipynb`
   6. `maps.ipynb`
   7. `cosmology.ipynb`

#### Running the python script

1. Create the python script from the jupyter notbooks. In `notebooks`:
  ```bash
  python config_convert.py
  ```

2. Run python script. In run directory:
  ```bash
   ipython /path/to/sp_validation/notebooks/validation.py
   ```

## Further post-processing

After the validation is run, further processing steps can be carried out using python scripts, as follows.

### Combine validation runs

Summary statistics created by validation runs of sub-areas of a survey can be combined to create joint summary statistics.
This is useful in cases where the galaxy catalogue of an entire survey is too large to process, and needs to be broken
down in smaller patches. This step provides global summary statistics from those patches.

Depending on the type of summary, their combination can be the sum (e.g. for number of objects), average, weighted average (e.g. for the additive bias),
the weighted average of the square (e.g. the ellipticity dispersion), the weighted variance (to combine variance estimates), or the weighted variance of the mean
(to combine mean variance estimates).

In a directory containing the subpatches as subdirectories, and within each their own `sp_output` results of the validation runs, type
```bash
/path/to/sp_validation/scripts/combine_results.py
```
This script creates a number of output files, including `R.txt` and `c.txt` with the combined multiplicative and additive biases, respectively.

### Create combined calibrated shear catalogue

After creating the combined results described above, the global calibration outputs can be used to create a combined, globally calibrated shear catalogue.
The calibration is obtained from the files `R.txt` and `c.txt` created above.

In the same directory containing the subpatches as above, type
```bash
/path/to/sp_validation/scripts/create_joint_shape_cat.py
```
It creates the joint output catalogue `joint.fits`.

### PSF - galaxy object-wise ellipticity leakage

The validation notebooks, in particular `psf_leakage.ipynb`, compute the leakage from PSF to galaxy ellipticity, as part of the global validation.
This can also be done with the stand-alone python script `scripts/leakage_object.py`.

For example, for a given patch, run
```bash
leakage_object.py -i sp_output/shape_catalog_extended_ngmix.fits -o leakage --hdu_psf 2 -v
```
to output plots and text files for the object-wise and scale-dependent leakage for that patch.

Leakage for the joint v1.0 catalogue can be computed via
```bash
leakage_object.py -i SP/unions_shapepipe_extended_2022_v1.0.fits -I SP/unions_shapepipe_psf_2022_v1.0.1.fits -o leakage -v
```
assuming `SP` is a link to the v1.0 ShapePipe data directory.
If this call was done in a subdirectory from where in `..` are the patch runs, joint plots of the scale-dependent leakage can be produced by
```bash
plot_leakage.py leakage/alpha_leakage_ngmix.txt ../P[1234567]/leakage/alpha_leakage_ngmix.txt`
```
This will read in the text files produces by the previous calls of `leakage_object.py`.

This script fits a consistent linear or quadratic model of the galaxy
ellipticity as function of PSF ellipticity. The best-fit parameters are written
in an output `json` file.

A summary table of the object-wise leakage linear parameters will be created by
`combine_results.py`, see above. This call will add the leakage from the joint
catalogue, assuming the results are stored in `joint/leakage`.

### PSF - galaxy scale-dependent ellipticity leakage

Use the script `scripts/leakage_scale.py`.

### Correct for PSF - galaxy ellipticity leakage

Experimentally we can apply the correction alpha * eps_PSF to the galaxy
elliptity.

First, `leakage_object.py` (see above) needs to be run.

Second, the best-fit result from that previous call of `leakage_object.py` is
read, applied to the extended shear catalogue, and the corrected ellipticity is
written in an output file.

```bash
apply_alpha.py`
```

Third, compute the shear correlation function and E-/B-mode statistics
using `treecorr`, with the notebook (or ipython script)
`notebooks/analt/xip_xim_v1_alpha_cor[.ipynb|.py]`.

Fourth, use the notebook `TBD` to plot the results. 

