# Calibration scripts

The calibrate stage: the scripts that turn a final `ShapePipe` output
catalogue into a science-ready, metacalibrated shear catalogue, and give its
selection a redshift distribution. Run them in order. See
`docs/source/post_processing.md` for the full prose.

| Step | Script | Does |
|------|--------|------|
| 1 | `extract_info.py` | Extract metacal + diagnostic info for one campaign; create pre-calibration shear catalogues. Configured via `params.py`. |
| 2 | `create_joint_comprehensive_cat.py` | Merge the per-campaign comprehensive catalogues into one joint catalogue (front-end of `catalog_builders.JointCat`). |
| 3 | `demo_apply_hsp_masks.py` | Add the structural and coverage (HealSparse) masks. |
| 4 | `calibrate_comprehensive_cat.py` | Galaxy selection + metacalibration. Uses the mask configs in `config/calibration/`. |
| 5 | `calibrate_nz.py` | SOM direct calibration: the tomographic n(z) file, each row's `TOM_BIN_ID` and n(z) realisations. |

`params.py` is the shared parameter template (paths, column names, survey
constants) imported by `extract_info.py`; copy and edit it per run.

> **v2.0 note:** ShapePipe v2 has no patches. Step 1 runs per campaign and
> step 2 merges a list of campaign catalogues (`final_cat_<campaign>.hdf5`).

## Redshift distribution (step 5)

The n(z) file is this stage's output. A `cosmo_val/cat_config.yaml` entry
names it in `shear.redshift_path` and its bin column in `tomo_bin_col`;
cosmo_val only reads it. The file has z in column zero and one density per
bin, each integrating to that bin's retained shear-weight fraction; their sum
has unit integral.

The recipe for v2 is `config/calibration/nz/v2.yaml`, written with data roots
(`config/data_roots.yaml`). Run inside the container on an allocation:

```bash
python scripts/calibration/calibrate_nz.py config/calibration/nz/v2.yaml --prepare --workers 32
```

`--prepare` builds the two inputs (a minute on 32 cores); later runs reuse
them. `--output DIRECTORY` redirects the products and `--figures DIRECTORY`
renders diagnostics. The products (`sp_validation.nz_pipeline`) are
`nz_tomo.txt`, `nz_nontomo.txt` (their sum), `tom_bin.parquet` (`TOM_BIN_ID`
and `ZBAND` row-aligned with the catalogue), `realisations.npz`,
`calibration_arrays.npz`, `som_model.npz` and `summary.json`.

**Sources** are the rows of the catalogue cosmo_val measures, with its `w_des`.
Sources and calibrators share one photometry: PhotoPipe's run on Stephen
Gwyn's DR6 detections (`UNIONS.<tile>_ugriz_photoz_ext.cat`, dereddened GAaP
ugriz, z2 and BPZ). Each catalogue row is found in the ShapePipe HATS
catalogue by exact position, and the PhotoPipe HATS catalogue beside it is
joined row for row on (tile, `NUMBER` = `SeqNr`) (`sp_validation.nz_inputs`).

**Calibrators** are the DEEP2 and DEEP3 spectra in the EGS field that
PhotoPipe matched to its own objects (`UNIONS.<tile>_ugriz_photoz_ext_specz.cat`,
surveys named by the redshift compilation beside it), one row per spectrum,
with complete ugriz. Stage them from
`arc:projects/unions/catalogues/unions/GAaP_photometry/UNIONS_DR6/UNIONS.<tile>/`
for the tiles around RA 215°, Dec 52.8°.

**Method.** A 101×101, 60-epoch batch SOM of all ten ugriz colours plus r,
standardized per feature and trained on the calibrators, grouped into 2,000
Ward groups. Bins are DR6 BPZ intervals [0.10, 0.37, 0.55, 0.76, 1.20), lower
edges included. Within a bin, a group is gold when it holds in-bin
calibrators and their mean spec-z agrees with its sources' weighted mean BPZ
within max(0.4, 5 NMAD of spec-z − BPZ); the test runs on every source of the
catalogue. Each spectrum carries its group's source shear weight divided by
the group's number of in-bin calibrators. Deposition on [0, 5] with Δz = 0.01
conserves mass and first moment. Canonical z takes precedence; detected z2
fills missing z after the recipe's offset, for sources and calibrators alike.

**Realisations.** With an `ensemble` section in the recipe, the run also
writes `realisations.npz`: `z` and `realisations` (n_draw, n_bin, n_z), each
bin unit-integral, every bin drawn together; `python -m
sp_validation.nz_ensemble calibration_arrays.npz -o realisations.npz` draws
from an existing run. A draw multiplies each calibrator's weight by
q_i = m_i (1 + δ(z_i)) and renormalises within its (bin, SOM group), so every
group keeps its source weight (`sp_validation.nz_ensemble`):

- *Calibrator shot noise*: m_i are multinomial bootstrap counts of the
  spectra. Bins are calibrated by disjoint calibrators (split by BPZ),
  so this term is independent between bins.
- *Sample variance of the EGS field*: δ(z) is a lognormal galaxy density
  along the line of sight, per Δz = 0.01 cell, with covariance from the
  nonlinear matter power seen through a 2° × 0.25° footprint (the DEEP2/3
  strip) times bias² (b = 1.3). It is shared by every bin and is what
  correlates them. Within-group renormalisation cancels δ wherever a group's
  calibrators share a redshift, which is how the SOM suppresses sample
  variance.

The draws are recentred on the calibration's n(z): renormalising positive
weights within groups is nonlinear, which moves the raw ensemble mean by a few
thousandths in ⟨z⟩: an estimator bias, not a redshift shift.

Held fixed: the SOM, its groups, gold masks and source weights. Left out:
photometric zero points and the z2 offset (shared by every bin), SOM
training stochasticity (on v2, seeds move ⟨z⟩ by up to 0.0015 in bin 4),
spectroscopic selection and incompleteness, spec-z errors, redshift-space
smearing of δ, and any bias evolution. The footprint and bias are
assumptions, and the δ term scales with b.

**Limits.** The calibrators are about 9,900 EGS DEEP2/3 spectra: no
field-to-field variance and no model of spectroscopic selection, so the n(z)
is for validation, null tests and mocks, not parameter constraints. It
describes only `TOM_BIN_ID > 0` rows; a non-tomographic measurement pairs
`nz_nontomo.txt` with those rows, not with the whole catalogue.
