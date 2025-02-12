## Post processing

Post processing steps carried out by the `sp_validation` package are:
- merging `shapepipe` output catalogues, e.g. processed by individual patches, into one or more joint catalogues;
- masking of objects using flags and criteria in `shapepipe` output catalogues and external (e.g. mask) files;  
- creating a galaxy sample by applying selection criteria, e.g. on SNR or size;  
- calibrating the shear estimates with the `metacalibration` method, using the measured shapes and
  metacal information (sheared measurements) output by `shapepipe`.

These steps are carried out as follows:

### Merge catalogues.

This is performed (currently both for pre- and post-v1.4.1 versions) with the series of notebooks
in `sp_validation/notebooks` or the `ipython` script `validation.py` generated thereof.

This script creates plots, diagnostics, and three shear catalogue FITS files:
- Basic catalogue containing
  positions, shapes (calibrated +  PSF-leakage corrected), weights (DES), magnitude, patch ID. Masking and galaxy selection are applied.
- Extended catalogue containing **in addition**
  uncalibrated shapes inverse-variance weights, shear response matrices, SNR, flux, size, PSF quantities. Masking and galaxy selection are applied.  
- Comprehensive catalogue containing **in addition**
  metacal information (measured sheared quantities), mask information (`shapepipe` pre-processing). Masking and galaxy selection is not applied.
  This catalogue does not contain calibrated shear estimates, since the calibration is carried out after applying masking and selection.


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
