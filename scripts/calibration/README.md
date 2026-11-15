# Calibration scripts

The catalogue calibration pipeline: the scripts that turn a final `ShapePipe`
output catalogue into a science-ready, metacalibrated shear catalogue. Run them
in order. See `docs/source/post_processing.md` for the full prose.

| Step | Script | Does |
|------|--------|------|
| 1 | `extract_info.py` | Extract metacal + diagnostic info for one campaign; create pre-calibration shear catalogues. Configured via `params.py`. |
| 2 | `create_joint_comprehensive_cat.py` | Merge the per-campaign comprehensive catalogues into one joint catalogue (front-end of `catalog_builders.JointCat`). |
| 3 | `demo_apply_hsp_masks.py` | Add the structural and coverage (HealSparse) masks. |
| 4 | `calibrate_comprehensive_cat.py` | Galaxy selection + metacalibration (`sp_validation.response.calibrate_catalogue`). Uses the mask configs in `config/calibration/`, whose `metacal` block must declare `weight_scheme: des` or `iv`. |

Step 4 writes the cut catalogue `shape_catalog_cut_<shape>.fits` and beside it:

- `<stem>_response.json`: R, its shear, selection and joint parts, `c_e`,
  `c_g`, the branch populations, and (with `TILE_ID`) the tile-jackknife
  covariance;
- `<stem>_sums.npz`: the per-tile response sums (`response.Sums`), from which
  `scripts/compute_m_bias_image_sims.py` measures image-simulation m;
- `<stem>_des_weights.fits`: the DES weight table (`response.write_table`),
  for `weight_scheme: des`.

A `metacal.weight_table` path makes step 4 weight with that table instead of
fitting one (image simulations use the data's table).

`params.py` is the shared parameter template (paths, column names, survey
constants) imported by `extract_info.py`; copy and edit it per run.

> **v2.0 note:** ShapePipe v2 has no patches. Step 1 runs per campaign and
> step 2 merges a list of campaign catalogues (`final_cat_<campaign>.hdf5`).
