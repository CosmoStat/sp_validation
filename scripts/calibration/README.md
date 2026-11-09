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
| 5 | `prepare_nz_photometry.py` | Attach DR6 ugriz + BPZ to the selected shear rows (n(z) input cache). |
| 6 | `calibrate_nz.py` | SOM direct calibration: the tomographic n(z) file and each row's `TOM_BIN_ID`. |

`params.py` is the shared parameter template (paths, column names, survey
constants) imported by `extract_info.py`; copy and edit it per run.

> **v2.0 note:** ShapePipe v2 has no patches. Step 1 runs per campaign and
> step 2 merges a list of campaign catalogues (`final_cat_<campaign>.hdf5`).

## Redshift distribution (steps 5–6)

The n(z) file is this stage's output. A `cosmo_val/cat_config.yaml` entry
names it in `shear.redshift_path` and its bin column in `tomo_bin_col`;
cosmo_val only reads it. The file has z in column zero and one density per
bin, each integrating to that bin's retained shear-weight fraction; their sum
has unit integral.

The recipe for v2 is `config/calibration/nz/v2_bound9.yaml`. Run inside the
container on an allocation:

```bash
python scripts/calibration/prepare_nz_photometry.py config/calibration/nz/v2_bound9.yaml
python scripts/calibration/calibrate_nz.py config/calibration/nz/v2_bound9.yaml --workers 16
```

Step 5 writes a cache that later calibrations reuse. Step 6 uses the frozen
SOM cells, groups and gold masks and recomputes group weights over the full
source selection; `--rebuild-model` retrains them from the inputs,
`--output DIRECTORY` redirects the products and `--figures DIRECTORY` renders
diagnostics. The recipe's input paths are candide's; to regenerate elsewhere,
copy the inputs and edit the paths. The frozen-delivery reproduction gate is
`scratch/cdaley/nz_som_reproduction/`.

**Photometry.** Step 5 matches each shear row to the nearest
`UNIONS.<tile>_SP_ugriz_photoz_ext.cat` object within 0.8″ in the same tile,
preserving row order. Those files are a PhotoPipe run on ShapePipe v1
detections, so for v2 shapes this is a positional match across two detection
runs, and the SOM was trained on that photometry. ShapePipe v2 and the
`UNIONS.<tile>_ugriz_photoz_ext.cat` files both number objects with Stephen
Gwyn's DR6 IDs and join exactly on (tile, `NUMBER` = `SeqNr`); moving the
recipe to that photometry is a recalibration (the two runs differ by
0.01–0.02 mag).

**Method.** A 101×101, 60-epoch batch SOM of all ten ugriz colours plus r,
standardized per feature, grouped into 2,000 Ward groups. Bins are DR6 BPZ
intervals [0.10, 0.37, 0.55, 0.76, 1.20), lower edges included. A gold group
has in-bin spectra and passes |⟨spec-z⟩ − ⟨BPZ⟩| ≤ max(0.4, 5 NMAD) on the
600-tile reference sample; groups empty there skip that check. Each spectrum
is weighted by its group's source shear weight divided by its number of
in-bin calibrators. Deposition on [0, 5] with Δz = 0.01 conserves mass and
first moment. Canonical z takes precedence; detected z2 fills missing z after
the recipe's offset. BPZ stays float64 to keep values on bin edges.

The calibration sample, `inputs/matched_calibration.npz`, holds the v1.4.6.3
sample's fiducial spectroscopic `Z_B` as `z`, DR6 BPZ `Z_B` as `photoz`, and
detected ugriz magnitudes as `mag`.

**Realisations.** With an `ensemble` section in the recipe, step 6 also
writes `realisations.npz`: `z` and `realisations` (n_draw, n_bin, n_z), each
bin unit-integral, every bin drawn together; `python -m
sp_validation.nz_ensemble calibration_arrays.npz -o realisations.npz` draws
from an existing run. A draw multiplies each calibrator's weight by
q_i = m_i (1 + δ(z_i)) and renormalises within its (bin, SOM group), so every
group keeps its source weight (`sp_validation.nz_ensemble`):

- *Calibrator shot noise*: m_i are multinomial bootstrap counts of the
  11,114 spectra. Bins are calibrated by disjoint calibrators (split by BPZ),
  so this term is independent between bins.
- *Sample variance of the EGS field*: δ(z) is a lognormal galaxy density
  along the line of sight, per Δz = 0.01 cell, with covariance from the
  nonlinear matter power seen through a 2° × 0.25° footprint (the DEEP2/3
  strip) times bias² (b = 1.3). It is shared by every bin and is what
  correlates them. Within-group renormalisation cancels δ wherever a group's
  calibrators share a redshift, which is how the SOM suppresses sample
  variance (3× in ⟨z⟩ on v2).

The draws are recentred on the calibration's n(z): renormalising positive
weights within groups is nonlinear, and on v2 it moves the raw ensemble mean
by up to +0.003 in ⟨z⟩ (0.75σ), an estimator bias, not a redshift shift.

Held fixed: the SOM, its groups, gold masks and source weights. Left out:
photometric zero points and the z2 offset (shared by every bin; the
`z2offset_lo/hi` products bracket the latter), SOM training stochasticity,
spectroscopic selection and incompleteness, spec-z errors, redshift-space
smearing of δ, and any bias evolution. The footprint and bias are
assumptions, and the δ term scales with b.

**Limits.** The calibrators are 11,114 EGS DEEP2/3 spectra: no
field-to-field variance and no model of spectroscopic selection, so the n(z)
is for validation, null tests and mocks, not parameter constraints. It
describes only `TOM_BIN_ID > 0` rows, so a non-tomographic run over all
objects pairs an all-object sample with a gold-only n(z). Source weights come
from the 22,200-tile bound-9 selection (191,016,178 rows), not the
19,940-tile catalogue cosmo_val measures.
