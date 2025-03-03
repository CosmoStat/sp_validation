## Science-ready catalogue production

Processing steps of `ShapePipe` output catalogues carried out by the `sp_validation` package to produce science-ready catalogues are:
1. Extract relevant information from a final `ShapePipe` output catalogue per patch; run basic diagnostic tests, create pre-calibration shear catalogues.
2. Merge pre-calibration catalogues created in the previous step, e.g. processed by individual patches, into one or more joint catalogues;
3. Apply external area and footprint masks. These are the "structural" and the coverage masks.  
4. Create calibrated galaxy shear catalogue. This step includes the tasks:  
   a. Mask objects using flags and criteria in `ShapePipe` output catalogues and external (e.g. mask) files;  
   b. Select a galaxy sample by applying selection criteria, e.g. on SNR or size;  
   c. Calibrate the shear estimates with the `metacalibration` method, using the measured shapes and metacal information (sheared measurements) output by `ShapePipe`.

These steps are carried out as follows:

### 1. Extract information, run basic diagnostics, create catalogues.

This is performed (currently both for pre- and post-v1.4.1 versions) with the series of notebooks
in `sp_validation/notebooks` or the `ipython` script `validation.py` generated thereof.

This script creates plots, diagnostics, and three shear catalogue FITS files:
- _Basic_ catalogue containing
  positions, shapes (calibrated +  PSF-leakage corrected), weights (DES), magnitude, patch ID. Masking and galaxy selection are applied.
- _Extended_ catalogue containing **in addition**
  uncalibrated shapes inverse-variance weights, shear response matrices, SNR, flux, size, PSF quantities. Masking and galaxy selection are applied.  
- _Comprehensive_ catalogue containing **in addition**
  metacal information (measured sheared quantities), mask information (`shapepipe` pre-processing). Masking and galaxy selection is not applied.
  This catalogue does not contain calibrated shear estimates, since the calibration is carried out after applying masking and selection.

This step is carried out per patch.

### 2. Merge catalogues

The patch-wise comprehensive catalogues extracted in the previous step are merged using the script `create_joint_comprehensive_cat.py`, which is a front-end
of the `sp_validation` library class `run_joint_cat:JointCat`.

### 3. Apply external masks

Code for this step is developed in the library file `run_calibrate_joint.py`.

### 4. Mask, select, and calibrate

The steps of masking, galaxy sample selection, and calibration are carried out jointly using the notebook
`calibrate_comprehensive_cat.ipynb`.

---

The following describes the pre-v1.4.2 method to create a joint, calibrated shear catalogue.

### Combine shear validation run output catalogues


### Combine shear validation run statistics

Summary statistics created by shear validation runs of sub-areas of a survey
can be combined to create joint summary statistics. This is useful in cases
where the galaxy catalogue of an entire survey is too large to process, and
needs to be broken down in smaller patches. This step provides global summary
statistics from those patches.

Depending on the type of summary, their combination can be the sum (e.g. for
number of objects), average, weighted average (e.g. for the additive bias), the
weighted average of the square (e.g. the ellipticity dispersion), the weighted
variance (to combine variance estimates), or the weighted variance of the mean
(to combine mean variance estimates).

In a directory containing the subpatches as subdirectories, and within each
their own output directory (`sp_output`by default in `params.py`) with results
of the validation runs, type
```bash
combine_results.py
```
This script creates a number of output files, including `R.txt` and `c.txt`
with the combined multiplicative and additive biases, respectively.

At the moment (ShapePipe catalogues v1.4.1 and older), this script needs to
be run to write summary statistics output files before creating the joint
shear catalogue with `create_joint_shape_cat.py`, see
[here](#create-combined-calibrated-shear-catalogue).


### Create combined calibrated shear catalogue

After creating the combined statistics results described above, the global
calibration outputs can be used to create a combined, globally calibrated shear
catalogue. The calibration is obtained from the files `R.txt` and `c.txt`
created above.

In the same directory containing the subpatches as above, type
```bash
create_joint_shape_cat.py
```
It creates the joint output catalogues
`{survey}_{pipeline}_{year}_v{version}.fits`
(e.g. `unions_shapepipe_2022_v1.4.fits`) and
`{survey}_{pipeline}_extended_{year}_v{version}.fits`.
