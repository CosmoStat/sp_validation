# Post-processing

## Science-ready catalogue production

Processing steps of `ShapePipe` output catalogues carried out by the `sp_validation` package to produce science-ready catalogues are:
1. Extract relevant information from a final `ShapePipe` output catalogue per campaign; run basic diagnostic tests, create pre-calibration shear catalogues.
2. Merge pre-calibration catalogues created in the previous step, e.g. processed as individual campaigns, into one or more joint catalogues;
3. Apply external area and footprint masks. These are the "structural" and the coverage masks.  
4. Create calibrated galaxy shear catalogue. This step includes the tasks:  
   a. Mask objects using flags and criteria in `ShapePipe` output catalogues and external (e.g. mask) files;  
   b. Select a galaxy sample by applying selection criteria, e.g. on SNR or size;  
   c. Calibrate the shear estimates with the `metacalibration` method, using the measured shapes and metacal information (sheared measurements) output by `ShapePipe`.

These steps are carried out as follows:

### 1. Extract information, run basic diagnostics, create catalogues.

This is performed (version > v1.4.1, < v2.0) with the python script `scripts/calibration/extract_info.py`.

This script creates three shear catalogues in FITS format:
- _Basic_ catalogue containing
  positions, shapes (calibrated +  PSF-leakage corrected), weights (DES), magnitude, campaign ID. Masking and galaxy selection are applied.  
- _Extended_ catalogue containing **in addition**
  uncalibrated shapes inverse-variance weights, shear response matrices, SNR, flux, size, PSF quantities. Masking and galaxy selection are applied.  
- _Comprehensive_ catalogue containing **in addition**
  metacal information (measured sheared quantities), mask information (`shapepipe` pre-processing). Masking and galaxy selection is not applied.
  This catalogue does not contain calibrated shear estimates, since the calibration is carried out after applying masking and selection.  
  This is the main output catalogue that will be processed further.

This step is carried out per campaign. Parameters have to be set via the python configuration file `params.py` (template at `scripts/calibration/params.py`).

### 2. Merge catalogues

The per-campaign comprehensive catalogues extracted in the previous step are merged using the script `scripts/calibration/create_joint_comprehensive_cat.py`, which is a front-end
of the `sp_validation` library class `catalog_builders:JointCat`.

### 3. Apply external masks

The structural and coverage masks are added with `scripts/calibration/demo_apply_hsp_masks.py` (built on the library file `run_calibrate_joint.py`).

### 4. Mask, select, and calibrate

The steps of masking, galaxy sample selection, and calibration are carried out jointly using the script
`scripts/calibration/calibrate_comprehensive_cat.py`.

Masking parameters have to be set via a configuration file `config_mask.yaml`. Examples can be found in `sp_validation/config/calibration`.
