# Stages

sp_validation turns ShapePipe and PhotoPipe outputs into cosmology in four
stages. Each one hands the next a file, and nothing downstream reaches back
past that file. This page maps the code as it is today and lists where it
does not yet match.

| Stage | Does | Code | Hands on |
|---|---|---|---|
| **Assemble** | Joins per-campaign ShapePipe outputs, star catalogues, structural masks and PhotoPipe photometry into one comprehensive catalogue | `scripts/calibration/` steps 1–3 (`extract_info.py`, `create_joint_comprehensive_cat.py`, `demo_apply_hsp_masks.py`); `catalog_builders.JointCat`, `ApplyHspMasks`; `scripts/merge_psf_cat.py`, `combine_hp_masks.py`, `fill_photoz_bands.py` | comprehensive catalogue (`final_cat_*.hdf5`), star catalogue, photometry joined row for row |
| **Calibrate** | Selection, metacal response and weights, PSF-leakage correction, multiplicative bias from image sims, and the redshift distribution | `scripts/calibration/` steps 4–6; `calibration.py`, `catalog_builders.CalibrateCat`, `nz_calibration.py`, `nz_photometry.py`, `nz_pipeline.py`; `config/calibration/` (`mask_*.yaml`, `nz/*.yaml`); `scripts/apply_alpha*.py`; `image_sims.py`, `workflow/rules/image_sims.smk` | a `cosmo_val/cat_config.yaml` entry: shear catalogue, PSF and star catalogues, footprint mask, n(z) file and bin column, blind |
| **Validate** | Two-point statistics, PSF systematics, B-mode and other nulls, covariance, all blinded | `src/sp_validation/cosmo_val/`, `cosmo_val/`, `workflow/rules/{cosmo_val,twopoint,covariance,masks}.smk`, `blinding.py`, `sacc_io.py` | `{version}.sacc`: concealed data vector, covariance and n(z) tracers; null-test verdicts |
| **Infer** | Likelihood, priors, sampler | `cosmo_inference/`, `workflow/rules/inference.smk` | chains |

**The calibrate → validate contract is the catalogue entry.** Everything
validation needs about a catalogue is named in its `cat_config.yaml` entry,
and validation reads only what is named there. A calibration that changes
(a new n(z), a new weight) changes the file the entry points to, or adds an
entry; it never changes validation code.

**Mocks run beside the stages.** GLASS mocks (`glass_mock.py`,
`scripts/glass_mock/`, `workflow/rules/glass_mock.smk`) are registered as
catalogue entries and go through validate exactly as data do. Image
simulations feed calibrate's multiplicative bias.

## Redshifts

The n(z) is calibrated, not validated:

1. **Assemble** gives each shape its PhotoPipe row. ShapePipe v2 and the
   `UNIONS.<tile>_ugriz_photoz_ext.cat` PhotoPipe run both number objects with
   Stephen Gwyn's DR6 IDs, so the join is exact on (tile, `NUMBER` = `SeqNr`)
   (`photopipe_merge.py`, branch `feat/photopipe-merge`).
2. **Calibrate** runs the SOM recipe (`config/calibration/nz/v2_bound9.yaml`,
   `scripts/calibration/` steps 5–6) and writes the n(z) file and each row's
   `TOM_BIN_ID`.
3. **Validate** reads that file through `shear.redshift_path` and the bins
   through `tomo_bin_col`.

## Where the code does not match yet

- **Three photometry joins.** `fill_photoz_bands.py` (ShapePipe v1, by row
  order against `_SP_ugriz` files), `nz_photometry.py` (v2 shapes, by
  position against the same `_SP_ugriz` files, which were run on v1
  detections) and `photopipe_merge.py` (v2, exact on Stephen's IDs). One join
  per ShapePipe version belongs in assemble; `nz_photometry.py` goes when the
  n(z) is recalibrated on the ID-joined photometry, which changes the
  magnitudes by 0.01–0.02.
- **Top-level `scripts/` mixes stages.** Assembly (`merge_psf_cat.py`,
  `combine_hp_masks.py`, `apply_hsp_masks.py`, `fill_photoz_bands.py`),
  calibration (`apply_alpha*.py`, `compute_m_bias_image_sims.py`,
  `get_nz_txt_files_from_fits.py`) and one-off diagnostics sit side by side.
  Moving them into `scripts/assembly/` and `scripts/calibration/` is
  mechanical.
- **`catalog_builders.py` holds both assemble (`JointCat`, `ApplyHspMasks`)
  and calibrate (`CalibrateCat`) classes.**
- **Only validate and infer are workflow rules.** Assemble and calibrate run
  as ordered scripts, apart from the image-sim bias. The n(z) recipe is the
  natural first calibrate-stage workflow (or ASTRA spec).
- **Paths are per machine.** Catalogue entries and recipes carry absolute
  paths for one cluster; the same entry does not resolve on another.
