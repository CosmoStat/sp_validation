# Calibration scripts

The catalogue calibration pipeline: the scripts that turn a final `ShapePipe`
output catalogue into a science-ready, metacalibrated shear catalogue. Run them
in order. See `docs/source/post_processing.md` for the full prose.

| Step | Script | Does |
|------|--------|------|
| 1 | `extract_info.py` | Extract metacal + diagnostic info per patch; create pre-calibration shear catalogues. Configured via `params.py`. |
| 2 | `create_joint_comprehensive_cat.py` | Merge the patch-wise comprehensive catalogues into one joint catalogue (front-end of `run_joint_cat.JointCat`). |
| 3 | `demo_apply_hsp_masks.py` | Add the structural and coverage (HealSparse) masks. |
| 4 | `calibrate_comprehensive_cat.py` | Galaxy selection + metacalibration. Uses the mask configs in `config/calibration/`. |

`params.py` is the shared parameter template (paths, column names, survey
constants) imported by `extract_info.py`; copy and edit it per run.

> **v2.0 note (Martin):** this clustering reflects the current (v1.4.x) reduction
> flow. v2.0 no longer has patches, so step 2 (and the per-patch structure of
> steps 1/3) will change.
