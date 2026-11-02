# Provisional v2 n(z)

`v2_bound9.yaml` records the four-bin SOM calibration recipe for the full-area bound-9 selection.
The output has z in column zero and four densities whose integrals are the retained shear-weight fractions; their sum has unit integral.
The n(z) describes only `TOM_BIN_ID > 0` objects, so a non-tomographic run over all objects pairs an all-object sample with a gold-only n(z).
Source weights come from the 22,200-tile selection (191,016,178 rows), rather than the 19,940-tile catalogue measured by cosmo_val (170.6M rows).
cosmo_val reads the saved file through `shear.redshift_path` and never runs calibration or loads photometry.
The `SP_v2.0` entry uses Nibi catalogue and n(z) paths and declares `host: nibi`.
The candide n(z) is `/n17data/cdaley/unions/analyses/shear_tomographic/nz_v2_full/calibration/nz_tomo.txt`.
At another site, set `shear.redshift_path` to the local copy.

This EGS-only calibration is for validation and null tests.
It has no independent-field variance or model of spectroscopic selection and does not support parameter constraints.

Run from this checkout inside the sp_validation container on a Slurm allocation:

```bash
python scripts/prepare_nz_photometry.py cosmo_val/nz/v2_bound9.yaml
python scripts/calibrate_nz.py cosmo_val/nz/v2_bound9.yaml --workers 16
```

Photometry preparation is a separate step; reuse its cache for subsequent calibrations.
The calibration uses frozen SOM cells, groups and gold masks, and computes group weights over the full source selection.
`--rebuild-model` retrains the SOM, Ward groups and gold masks from the inputs.
`--output DIRECTORY` selects an output directory; `--figures DIRECTORY` renders diagnostics.
The frozen-delivery reproduction gate lives in `scratch/cdaley/nz_som_reproduction/`.

The calibration sample, `inputs/matched_calibration.npz`, contains the v1.4.6.3 sample's fiducial spectroscopic `Z_B` as `z`, DR6 BPZ `Z_B` as `photoz`, and detected ugriz magnitudes as `mag`.
The selected shear Parquet and its provenance are `inputs/bound9_full_selection.parquet` and `inputs/selection.json` in the analysis directory named by the recipe.
Photometry matching uses the nearest DR6 position within 0.8 arcsec in the same tile and preserves source row order.
ShapePipe `NUMBER` identifies the source within its tile.
BPZ stays float64 to preserve values on bin edges.

The recipe fixes a 101×101, 60-epoch batch SOM of all ten ugriz colours plus r magnitude, standardized per feature, and 2,000 Ward groups.
BPZ intervals [0.10, 0.37, 0.55, 0.76, 1.20] include lower edges and exclude upper edges.
Each gold group has in-bin spectra and passes the source-BPZ versus mean-spec-z threshold `max(0.4, 5 NMAD)` on the 600-tile reference sample.
Groups empty in that reference skip the BPZ-versus-spec-z check.
The source shear-weight sum divided by the number of in-bin calibrators weights each spectrum in its group.
Linear deposition on [0, 5] with Δz=0.01 conserves mass and first moment under trapezoidal integration.
Canonical z takes precedence; detected z2 fills missing canonical z after the offset in the recipe.
