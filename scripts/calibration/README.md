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
the group's number of in-bin calibrators. The spectra are deposited linearly
onto the recipe's grid ([0, 5] in steps of 0.05), which conserves each bin's
mass and mean and broadens it by about step²/6 in variance. The step sets the
delivered resolution: finer grids resolve the line-of-sight structure of the
single calibration field, which is not a property of the source population.
Against a 0.01 grid, 0.05 moves the ξ± theory (12–83′) by Δχ² < 0.01 without
intrinsic alignments and 0.06 with A_IA = 1, and leaves ⟨z⟩ unchanged.
Canonical z takes precedence; detected z2
fills missing z after the recipe's offset, for sources and calibrators alike.

**Realisations.** With an `ensemble` section in the recipe, the run also
writes `realisations.npz`: `z` and `realisations` (n_draw, n_bin, n_z), each
bin unit-integral, every bin drawn together, plus `ensemble_inputs.npz`, what
the draws need from the sources. `python -m sp_validation.nz_ensemble <run
directory> -o realisations.npz [--settings ensemble.yaml]` draws from an
existing run. A draw perturbs the calibration's inputs and recalibrates: each
in-bin spectrum of a gold group carries the group's source weight S_kg times
q_i / Σ_group q_j (`sp_validation.nz_ensemble`). Each key of the `ensemble`
section switches one term on.

Statistical terms, through q_i = m_i (1 + δ(z_i)):

- `bootstrap`: m_i are multinomial bootstrap counts of the spectra. Bins are
  calibrated by disjoint spectra (split by BPZ), so this term is independent
  between bins.
- `density`: δ(z) is a lognormal galaxy density along the line of sight,
  per grid cell, with covariance from the nonlinear matter power seen through
  a 2° × 0.25° footprint (the DEEP2/3 strip) times bias² (b = 1.3). It is
  shared by every bin. Within-group renormalisation cancels δ wherever a
  group's spectra share a redshift, which is how the SOM suppresses sample
  variance.

Systematic terms:

- `zero_points`: a zero-point error of EGS against the wide field. The
  spectra trained the SOM, so its cells would have moved with them; the error
  is applied as an offset of every source against the SOM, which moves source
  weight between groups, and as a shift of each spectrum's BPZ along its local
  BPZ gradient (a plane through its 64 nearest spectra), which can move it
  between bins. Offsetting the spectra against the fixed SOM instead would
  barely move them: they sit at the cell centres they trained, and a 0.02 mag
  offset changes the group of 0.8% of them. Sizes are measured with stellar-
  locus regression on the 46,700 stars (spec-z < 0.002, 17 < r < 21) of
  PhotoPipe's DR6 spec-z match: the colour offset of each 3.4 deg² patch (about
  the EGS footprint) at |b| > 30° scatters by 0.022, 0.020, 0.012 and 0.021 mag
  in u−g, g−r, r−i and i−z, less the patches' own noise. The grey term,
  0.01 mag, is the Pan-STARRS 3π uniformity the DR6 griz calibration inherits.
  EGS itself sits at +0.009, +0.035, −0.013 and +0.055 (± 0.009, 0.005, 0.004,
  0.005) off the survey locus; its i−z is redder than 99% of patches.
- `z2_offset`: an error in the constant that maps z2 onto z, applied to the z
  band of every source measured through z2 (a third of the gold weight; the
  spectra have none). σ = 0.02 mag: stars in z2-only areas sit 0.006–0.017 mag
  off the canonical i−z locus after the mapping, and galaxies may carry a
  colour term the stars don't.
- `specz_failure`: each spectrum's redshift is, with probability 0.01,
  replaced by the redshift of a random spectrum. PhotoPipe's compilation
  (2026-10-19) gives every DEEP2/3 redshift confidence 0.99. Newman et al.
  (2013) measure 0.75% (Q = 3) and 0.15% (Q = 4) failures from repeats, so
  0.01 is conservative.
- `nmad_factor`: the gold-QC factor drawn uniformly in [4, 6] around the
  fiducial 5 (threshold 0.40–0.56 with the 0.4 floor); gold is re-derived at
  that factor, so the bins' source selection moves with it. The other terms
  keep the catalogue's gold selection.

Source tables respond to the offsets linearly (central differences at
±0.02 mag; checked against exact reassignment at 0.05 mag). The draws are
recentred on the calibration's n(z): the calibration is nonlinear in its
inputs, which moves the raw ensemble mean by up to 0.005 in ⟨z⟩. Most of it
is estimator bias; spec-z failures alone pull bin 1 up by 0.004 toward the
calibrators' mean redshift, which is what random failures would do.

Held fixed: the SOM and its groups. Left out: spectroscopic selection and
incompleteness (DEEP2's [OII] leaves the spectral window above z ≈ 1.4, so
spectra at higher redshift are missing, not wrong), method bias calibrated on
simulations, SOM training stochasticity (on v2, seeds move ⟨z⟩ by up to
0.0015 in bin 4), redshift-space smearing of δ, and bias evolution.

**Limits.** The calibrators are about 9,900 EGS DEEP2/3 spectra: no
field-to-field variance and no model of spectroscopic selection, so the n(z)
is for validation, null tests and mocks, not parameter constraints. It
describes only `TOM_BIN_ID > 0` rows; a non-tomographic measurement pairs
`nz_nontomo.txt` with those rows, not with the whole catalogue.
