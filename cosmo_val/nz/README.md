# Provisional v2 n(z)

`v2_bound9.yaml` records the calibration recipe for the four-bin, full-area bound-9 selection.
The output is a text file with z in column zero and four densities whose integrals are the retained shear-weight fractions.
Their sum has unit integral and describes only objects with `TOM_BIN_ID > 0`.
cosmo_val reads this file through `shear.redshift_path`; it never loads the photometry or runs the calibration.
The `SP_v2.0` entry in `cat_config.yaml` uses the existing Nibi catalogue and n(z) paths.
Its `nz.recipe` points here for regeneration; it is passive metadata for the standalone calibrator.
The equivalent verified file for candide is `/n17data/cdaley/unions/analyses/shear_tomographic/nz_v2_full/calibration/nz_tomo.txt`.
When copying the file to another site, set that site's `shear.redshift_path` to the local copy.

This is an EGS-only working calibration for validation and null tests.
It has no independent-field variance or model of spectroscopic selection, and does not support parameter constraints.
The photo-z group's calibration can replace the method without changing the consumer file format.

Run from this checkout inside the sp_validation container on a Slurm allocation:

```bash
python scripts/calibrate_nz.py cosmo_val/nz/v2_bound9.yaml --workers 16 --verify
```

The default uses the frozen SOM, groups and gold masks from the 600-tile reference, and recalibrates the group weights over the full selection.
`--rebuild-model` retrains the map, Ward groups and gold masks from their durable inputs.
`--prepare` first attaches DR6 photometry to the selected shear objects; `scripts/prepare_nz_photometry.py` also runs this step separately.
Both commands use the same recipe.
`--output DIRECTORY` chooses a separate output directory and `--figures DIRECTORY` renders the diagnostics.
Verification compares the delivered means and every sidecar bin ID, and round-trips the actual cosmo_val reader and SACC NZ tracers.

The calibration sample is `inputs/matched_calibration.npz`, frozen under the analysis directory named in the recipe.
It contains the v1.4.6.3 calibration sample's fiducial spectroscopic `Z_B` as `z`, matched DR6 `Z_B` as `photoz`, and detected DR6 ugriz magnitudes as `mag`.
No alternative spectroscopic-redshift columns enter the calculation.
The shear selection and its provenance are `inputs/bound9_full_selection.parquet` and `inputs/selection.json` in the same directory.
Photometry matching uses the nearest DR6 position within 0.8 arcsec in the same tile, preserving the selected-catalogue row order.
ShapePipe `NUMBER` is an output key and is not a DR6 row identifier.
BPZ stays float64 so values on a bin edge retain their assigned interval.

The recipe fixes a 101×101, 60-epoch batch SOM of all ten ugriz colours plus r magnitude, standardized per feature, and 2,000 Ward groups.
The BPZ edges are [0.10, 0.37, 0.55, 0.76, 1.20], lower inclusive and upper exclusive.
Each gold group has in-bin spectra and passes the source-BPZ versus mean-spec-z threshold `max(0.4, 5 NMAD)` derived on the 600-tile reference.
The full-area source shear-weight sum divided by the number of in-bin calibrators weights each spectrum in its group.
Linear deposition on [0, 5] with Δz=0.01 conserves mass and first moment under trapezoidal integration.
Canonical z takes precedence; detected z2 fills missing canonical z after the exact scalar offset in the recipe.
The saved model and reference source sample make the frozen gold masks reproducible without depending on a scratch directory.
