# Stages

sp_validation turns ShapePipe and PhotoPipe outputs into cosmology in four
stages. Each one hands the next a file, and nothing downstream reaches back
past that file. This page maps the code as it is today and lists where it
does not yet match.

| Stage | Does | Code | Hands on |
|---|---|---|---|
| **Assemble** | Joins per-campaign ShapePipe outputs, star catalogues, structural masks and PhotoPipe photometry into one comprehensive catalogue | `scripts/calibration/` steps 1–3 (`extract_info.py`, `create_joint_comprehensive_cat.py`, `demo_apply_hsp_masks.py`); the v2 PhotoPipe join and the v2 HATS collection (`scripts/assembly/`, `photopipe_merge.py`, `object_id.py`, `hats_collection.py`); `catalog_builders.JointCat`, `ApplyHspMasks`; `scripts/merge_psf_cat.py`, `combine_hp_masks.py`, `fill_photoz_bands.py` | comprehensive catalogue (`final_cat_*.hdf5`, or for v2 a HATS collection: shapes, PhotoPipe extension and `OBJECT_ID` index), star catalogue, photometry joined row for row |
| **Calibrate** | Selection, metacal response and weights, PSF-leakage correction, multiplicative bias from image sims, and the redshift distribution | `scripts/calibration/` steps 4–5; `calibration.py`, `catalog_builders.CalibrateCat`, `nz_inputs.py`, `nz_calibration.py`, `nz_pipeline.py`, `nz_ensemble.py`; `config/calibration/` (`mask_*.yaml`, `nz/*.yaml`); `scripts/apply_alpha*.py`; `image_sims.py`, `workflow/rules/image_sims.smk` | a `cosmo_val/cat_config.yaml` entry: shear catalogue, PSF and star catalogues, footprint mask, n(z) file and bin column, blind |
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

1. **Assemble** gives each shape its PhotoPipe row (`scripts/assembly/`). ShapePipe v2 and the
   `UNIONS.<tile>_ugriz_photoz_ext.cat` PhotoPipe run both number objects with
   Stephen Gwyn's DR6 IDs, so the join is exact on (tile, `NUMBER` = `SeqNr`).
2. **Calibrate** runs the SOM recipe (`config/calibration/nz/v2.yaml`,
   `scripts/calibration/` step 5) on that photometry and the spectra
   PhotoPipe matched to the same objects, and writes the n(z) file, its
   realisations and each row's `TOM_BIN_ID`.
3. **Validate** reads the file through `shear.redshift_path` and the bins
   through `tomo_bin_col`.

## Paths across machines

Configs name data by root, not by machine. `config/data_roots.yaml` lists,
for each root, the directory that holds it on each cluster; a config writes
`${nz_v2}/nz_tomo.txt`, and `sp_validation.data_roots.load_yaml` replaces
the root with the first of its directories that exists. `SPV_ROOT_<NAME>`
points a root at a copy the file doesn't list. A root found nowhere stays
written as `${name}`, so the path fails when opened and names what is missing.
Every loader of `cosmo_val/cat_config.yaml` (cosmo_val, the workflow,
`blinding.py`, the dashboard) goes through `load_yaml`, so an entry runs
wherever its roots resolve. A root's directories share one relative layout.

## Where the code does not match yet

- **ShapePipe v1 keeps its own photometry join.** `fill_photoz_bands.py`
  joins v1 by row order against the `_SP_ugriz` PhotoPipe files, which ran on
  v1 detections. v2's join (`scripts/assembly/`) is exact on Stephen's IDs.
- **Top-level `scripts/` mixes stages.** Assembly (`merge_psf_cat.py`,
  `combine_hp_masks.py`, `apply_hsp_masks.py`, `fill_photoz_bands.py`),
  calibration (`apply_alpha*.py`, `compute_m_bias_image_sims.py`,
  `get_nz_txt_files_from_fits.py`) and one-off diagnostics sit side by side.
  Moving them into `scripts/assembly/` and `scripts/calibration/` is
  mechanical; only the v2 collection lives in `scripts/assembly/` so far.
- **`catalog_builders.py` holds both assemble (`JointCat`, `ApplyHspMasks`)
  and calibrate (`CalibrateCat`) classes.**
- **Only validate and infer are workflow rules.** Assemble and calibrate run
  as ordered scripts, apart from the image-sim bias.
- **Most catalogue entries carry candide paths.** Only `SP_v2.0` is written
  with roots; the others move as their data land on a reachable machine.
